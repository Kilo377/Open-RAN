"""Unified feasibility-region projection for full-form orchestration plans."""

from __future__ import annotations

from dataclasses import replace

from models.intent_orchestration import (
    A1GuidanceProfile,
    ArbitrationProfileRef,
    FeasibilityConstraintSummary,
    OrchestrationCandidatePlan,
    QoSConstraintSpec,
    QoSProtectionProfile,
)
from models.observation import Observation
from ric.near_rt.xapp_variant_catalog import canonical_xapp_category
from ric.non_rt.intent_orchestration_helpers import (
    default_exploration_health_gate,
    default_forbidden_combinations,
    default_hard_safety_envelope,
)


_RESOURCE_COST_LIBRARY: dict[str, dict[str, float]] = {
    "mobility_predictor_xapp": {"cpu": 0.9, "mem": 0.8},
    "traffic_steering_xapp": {"cpu": 1.1, "mem": 1.0},
    "power_saving_xapp": {"cpu": 0.7, "mem": 0.7},
    "spectrum_optimizer_xapp": {"cpu": 1.2, "mem": 1.0},
    "latency_scheduler_xapp": {"cpu": 1.3, "mem": 1.1},
    "massive_mimo_beamformer_xapp": {"cpu": 1.5, "mem": 1.3},
    "ran_slicing_manager_xapp": {"cpu": 1.0, "mem": 0.9},
    "urllc_guard_xapp": {"cpu": 1.1, "mem": 1.0},
    "cell_sleep_xapp": {"cpu": 0.8, "mem": 0.8},
}

_DEFAULT_MUTEX_CATEGORY_PAIRS: tuple[tuple[str, str], ...] = (
    ("cell_sleep_xapp", "latency_scheduler_xapp"),
    ("cell_sleep_xapp", "urllc_guard_xapp"),
)

_DEFAULT_DEPENDENCY_CATEGORY_PAIRS: tuple[tuple[str, str], ...] = (
    ("cell_sleep_xapp", "power_saving_xapp"),
    ("traffic_steering_xapp", "mobility_predictor_xapp"),
)

_ETA_PROFILE_LIBRARY: dict[str, A1GuidanceProfile] = {
    "eta:balanced": A1GuidanceProfile(
        profile_name="eta:balanced",
        goal_focus="balanced",
        reward_term_bias={
            "throughput_gain": 1.0,
            "queue_reduction": 1.0,
            "bler_reduction": 1.0,
            "handover_risk_reduction": 1.0,
            "power_cost_penalty": 1.0,
        },
        field_priority_hints={},
        safety_rule_bias=["rule:hard_safety_envelope", "rule:exploration_health_gate"],
        action_region_bias="balanced",
        defaulted=False,
    ),
    "eta:throughput_push": A1GuidanceProfile(
        profile_name="eta:throughput_push",
        goal_focus="throughput",
        reward_term_bias={
            "throughput_gain": 1.25,
            "queue_reduction": 1.05,
            "bler_reduction": 0.95,
            "handover_risk_reduction": 0.90,
            "power_cost_penalty": 0.90,
        },
        field_priority_hints={
            "beam_candidate_preference": 1.25,
            "power_cap_dbm": 1.10,
            "scheduler_policy_hint": 1.05,
        },
        safety_rule_bias=["rule:hard_safety_envelope", "rule:forbidden_combinations"],
        action_region_bias="throughput_expansion",
        defaulted=False,
    ),
    "eta:reliability_guarded": A1GuidanceProfile(
        profile_name="eta:reliability_guarded",
        goal_focus="reliability",
        reward_term_bias={
            "throughput_gain": 0.90,
            "queue_reduction": 1.00,
            "bler_reduction": 1.30,
            "handover_risk_reduction": 1.05,
            "power_cost_penalty": 1.00,
        },
        field_priority_hints={
            "latency_bias": 1.20,
            "scheduler_policy_hint": 1.10,
            "beam_candidate_preference": 0.95,
        },
        safety_rule_bias=[
            "rule:hard_safety_envelope",
            "rule:exploration_health_gate",
            "rule:degraded_network_freeze",
        ],
        action_region_bias="guardrail_tight",
        defaulted=False,
    ),
    "eta:mobility_stable": A1GuidanceProfile(
        profile_name="eta:mobility_stable",
        goal_focus="mobility",
        reward_term_bias={
            "throughput_gain": 0.95,
            "queue_reduction": 1.00,
            "bler_reduction": 1.05,
            "handover_risk_reduction": 1.25,
            "power_cost_penalty": 1.00,
        },
        field_priority_hints={
            "mobility_bias_db": 1.25,
            "measurement_event_offset_db": 1.15,
            "steering_hysteresis_db": 1.10,
        },
        safety_rule_bias=["rule:hard_safety_envelope", "rule:forbidden_combinations"],
        action_region_bias="mobility_guarded",
        defaulted=False,
    ),
}

_CHI_PROFILE_LIBRARY: dict[str, ArbitrationProfileRef] = {
    "chi:balanced_weighted": ArbitrationProfileRef(
        profile_name="chi:balanced_weighted",
        composition_mode="parameter_tuning",
        arbitration_mode="weighted_merge",
        merge_scope="field",
        override_order=[],
        defaulted=False,
        notes=["Balanced weighted merge across selected xApp heads."],
    ),
    "chi:guardrail_first": ArbitrationProfileRef(
        profile_name="chi:guardrail_first",
        composition_mode="parameter_tuning",
        arbitration_mode="guardrail_first",
        merge_scope="field",
        override_order=["urllc_guard_xapp", "latency_scheduler_xapp", "ran_slicing_manager_xapp"],
        defaulted=False,
        notes=["Protect reliability-oriented heads before capacity-oriented heads."],
    ),
    "chi:priority_scheduler": ArbitrationProfileRef(
        profile_name="chi:priority_scheduler",
        composition_mode="parameter_tuning",
        arbitration_mode="priority_override",
        merge_scope="field",
        override_order=["latency_scheduler_xapp", "spectrum_optimizer_xapp", "massive_mimo_beamformer_xapp"],
        defaulted=False,
        notes=["Prefer scheduler-dominant categories when conflicts remain after projection."],
    ),
}


class OrchestrationFeasibilityProjector:
    """Projects requested orchestration candidates into a unified feasible region G_k."""

    def __init__(
        self,
        *,
        max_selected_xapps: int,
        max_cpu_cost: float = 6.0,
        max_mem_cost: float = 5.5,
        mutex_category_pairs: list[list[str]] | None = None,
        dependency_category_pairs: list[list[str]] | None = None,
    ) -> None:
        self.max_selected_xapps = max(1, int(max_selected_xapps))
        self.max_cpu_cost = max(0.5, float(max_cpu_cost))
        self.max_mem_cost = max(0.5, float(max_mem_cost))
        self.mutex_category_pairs = self._normalize_pairs(
            mutex_category_pairs,
            fallback_pairs=_DEFAULT_MUTEX_CATEGORY_PAIRS,
        )
        self.dependency_category_pairs = self._normalize_pairs(
            dependency_category_pairs,
            fallback_pairs=_DEFAULT_DEPENDENCY_CATEGORY_PAIRS,
        )

    def _normalize_pairs(
        self,
        pairs: list[list[str]] | None,
        *,
        fallback_pairs: tuple[tuple[str, str], ...],
    ) -> list[tuple[str, str]]:
        normalized: list[tuple[str, str]] = []
        raw_pairs = list(pairs or [])
        if len(raw_pairs) == 0:
            raw_pairs = [list(item) for item in fallback_pairs]
        for raw_pair in raw_pairs:
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                continue
            left = canonical_xapp_category(str(raw_pair[0] or "").strip())
            right = canonical_xapp_category(str(raw_pair[1] or "").strip())
            if left == "" or right == "" or left == right:
                continue
            pair = (left, right)
            if pair not in normalized:
                normalized.append(pair)
        return normalized

    def _goal_focus(self, goal) -> str:
        kpi_names = {
            str(getattr(target, "kpi_name", "") or "").strip().lower()
            for target in list(getattr(goal, "kpi_targets", []) or [])
            if str(getattr(target, "kpi_name", "") or "").strip() != ""
        }
        if "handover_risk" in kpi_names:
            return "mobility"
        if "power_cost" in kpi_names:
            return "energy"
        if "avg_bler" in kpi_names and "total_throughput_mbps" not in kpi_names:
            return "reliability"
        if "total_throughput_mbps" in kpi_names and "avg_bler" in kpi_names:
            return "throughput_reliability"
        if "total_throughput_mbps" in kpi_names:
            return "throughput"
        return "balanced"

    def _category_priority(self, category_name: str, *, goal_focus: str, risk_posture: str) -> float:
        category_name = canonical_xapp_category(category_name)
        base_score = {
            "massive_mimo_beamformer_xapp": 0.70,
            "spectrum_optimizer_xapp": 0.68,
            "traffic_steering_xapp": 0.66,
            "latency_scheduler_xapp": 0.64,
            "urllc_guard_xapp": 0.69,
            "mobility_predictor_xapp": 0.60,
            "ran_slicing_manager_xapp": 0.58,
            "power_saving_xapp": 0.52,
            "cell_sleep_xapp": 0.45,
        }.get(category_name, 0.50)
        focus_bonus = {
            "throughput": {
                "massive_mimo_beamformer_xapp": 0.30,
                "spectrum_optimizer_xapp": 0.24,
                "traffic_steering_xapp": 0.10,
            },
            "throughput_reliability": {
                "massive_mimo_beamformer_xapp": 0.20,
                "urllc_guard_xapp": 0.22,
                "spectrum_optimizer_xapp": 0.14,
                "latency_scheduler_xapp": 0.12,
            },
            "reliability": {
                "urllc_guard_xapp": 0.30,
                "latency_scheduler_xapp": 0.18,
                "ran_slicing_manager_xapp": 0.10,
            },
            "mobility": {
                "mobility_predictor_xapp": 0.30,
                "traffic_steering_xapp": 0.24,
            },
            "energy": {
                "power_saving_xapp": 0.26,
                "cell_sleep_xapp": 0.22,
            },
        }
        score = base_score + float(focus_bonus.get(goal_focus, {}).get(category_name, 0.0))
        if risk_posture == "conservative" and category_name in {"urllc_guard_xapp", "latency_scheduler_xapp"}:
            score += 0.10
        if risk_posture == "aggressive" and category_name in {"massive_mimo_beamformer_xapp", "spectrum_optimizer_xapp"}:
            score += 0.08
        return float(score)

    def _preferred_variant_name(
        self,
        category_name: str,
        *,
        available_xapp_names: list[str],
    ) -> str | None:
        requested_name = str(category_name or "").strip()
        available_set = {str(xapp_name) for xapp_name in list(available_xapp_names or [])}
        if requested_name in available_set:
            return requested_name
        category_name = canonical_xapp_category(requested_name)
        candidates = [
            str(xapp_name)
            for xapp_name in list(available_xapp_names or [])
            if canonical_xapp_category(str(xapp_name or "").strip()) == category_name
        ]
        if len(candidates) == 0:
            return None
        for preferred_token in ("balanced", "strict", "capacity_push", "quality_first"):
            for candidate_name in candidates:
                if preferred_token in candidate_name:
                    return candidate_name
        return candidates[0]

    def _resource_cost(self, xapp_name: str) -> tuple[float, float]:
        category_name = canonical_xapp_category(xapp_name)
        spec = _RESOURCE_COST_LIBRARY.get(category_name, {})
        return float(spec.get("cpu", 1.0) or 1.0), float(spec.get("mem", 1.0) or 1.0)

    def _summed_costs(self, selected_xapps: list[str]) -> tuple[float, float]:
        total_cpu = 0.0
        total_mem = 0.0
        for xapp_name in list(selected_xapps or []):
            cpu_cost, mem_cost = self._resource_cost(xapp_name)
            total_cpu += float(cpu_cost)
            total_mem += float(mem_cost)
        return float(total_cpu), float(total_mem)

    def _remove_lowest_priority(
        self,
        selected_xapps: list[str],
        *,
        goal_focus: str,
        risk_posture: str,
        protected_categories: set[str] | None = None,
    ) -> list[str]:
        protected_categories = set(protected_categories or set())
        ranked = sorted(
            list(selected_xapps or []),
            key=lambda xapp_name: (
                self._category_priority(
                    canonical_xapp_category(xapp_name),
                    goal_focus=goal_focus,
                    risk_posture=risk_posture,
                ),
                str(xapp_name),
            ),
        )
        for candidate_name in ranked:
            if canonical_xapp_category(candidate_name) not in protected_categories:
                return [item for item in selected_xapps if item != candidate_name]
        return list(selected_xapps[:-1])

    def _build_eta_profile(self, profile_name: str, *, goal, selected_xapps: list[str]) -> A1GuidanceProfile:
        template = replace(
            _ETA_PROFILE_LIBRARY.get(profile_name, _ETA_PROFILE_LIBRARY["eta:balanced"])
        )
        field_priority_hints = dict(getattr(template, "field_priority_hints", {}) or {})
        selected_categories = {
            canonical_xapp_category(xapp_name)
            for xapp_name in list(selected_xapps or [])
            if str(xapp_name).strip() != ""
        }
        if "traffic_steering_xapp" in selected_categories:
            field_priority_hints.setdefault("traffic_steering_preference", 1.08)
        if "mobility_predictor_xapp" in selected_categories:
            field_priority_hints.setdefault("measurement_event_offset_db", 1.05)
        template.field_priority_hints = field_priority_hints
        template.selected_categories = sorted(selected_categories)
        template.risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced")
        return template

    def _derive_qos_protection_profile(
        self,
        *,
        goal,
        observation: Observation,
        selected_xapps: list[str],
    ) -> QoSProtectionProfile:
        kpi_targets = list(getattr(goal, "kpi_targets", []) or [])
        strictness = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()
        constraints: list[QoSConstraintSpec] = []
        required_categories: list[str] = []
        selected_categories = {
            canonical_xapp_category(xapp_name)
            for xapp_name in list(selected_xapps or [])
            if str(xapp_name).strip() != ""
        }
        for target in kpi_targets:
            kpi_name = str(getattr(target, "kpi_name", "") or "").strip().lower()
            target_value = getattr(target, "target_value", None)
            if kpi_name == "avg_bler" and target_value is not None:
                constraints.append(
                    QoSConstraintSpec(
                        metric_name="avg_bler",
                        semantic_type="min",
                        threshold_value=float(target_value),
                        outage_budget=0.05 if strictness == "conservative" else 0.10,
                        scope="cluster",
                        notes=["Reliability guard derived from KPI target."],
                    )
                )
                required_categories.extend(["urllc_guard_xapp", "latency_scheduler_xapp"])
            if kpi_name in {"queue_bytes", "total_queue_bytes"} and target_value is not None:
                constraints.append(
                    QoSConstraintSpec(
                        metric_name="queue_bytes",
                        semantic_type="min",
                        threshold_value=float(target_value),
                        outage_budget=0.10,
                        scope="cluster",
                        notes=["Backlog QoS guard derived from queue KPI target."],
                    )
                )
            if kpi_name in {"delay_ms", "scheduler_hol_delay_ms", "demand_hol_delay_ms"} and target_value is not None:
                constraints.append(
                    QoSConstraintSpec(
                        metric_name="scheduler_hol_delay_ms",
                        semantic_type="min",
                        threshold_value=float(target_value),
                        outage_budget=0.05,
                        scope="cluster",
                        notes=["Latency outage guard derived from delay KPI target."],
                    )
                )
                required_categories.extend(["latency_scheduler_xapp", "urllc_guard_xapp"])
            if kpi_name == "handover_risk" and target_value is not None:
                constraints.append(
                    QoSConstraintSpec(
                        metric_name="handover_risk",
                        semantic_type="min",
                        threshold_value=float(target_value),
                        outage_budget=0.10,
                        scope="cluster",
                        notes=["Mobility QoS guard derived from handover KPI target."],
                    )
                )
                required_categories.extend(["mobility_predictor_xapp", "traffic_steering_xapp"])
        if len(constraints) == 0:
            avg_bler = float(getattr(observation, "avg_bler", 0.0) or 0.0)
            if avg_bler > 0.08 or strictness == "conservative":
                constraints.append(
                    QoSConstraintSpec(
                        metric_name="avg_bler",
                        semantic_type="min",
                        threshold_value=max(0.08, avg_bler),
                        outage_budget=0.10,
                        scope="cluster",
                        notes=["Default reliability guard from current observation."],
                    )
                )
                required_categories.extend(["urllc_guard_xapp"])
        return QoSProtectionProfile(
            profile_id=f"qos:{str(getattr(goal, 'goal_id', '') or '')}",
            strictness=strictness if strictness != "" else "balanced",
            constraints=constraints,
            required_categories=sorted(
                {
                    category_name
                    for category_name in required_categories
                    if category_name not in {"", *selected_categories}
                }
            ),
            generated_by="feasibility_projector",
        )

    def _qos_admissibility_projection(
        self,
        *,
        selected_xapps: list[str],
        qos_profile: QoSProtectionProfile,
        goal_focus: str,
        risk_posture: str,
        available_xapp_names: list[str],
    ) -> tuple[list[str], list[str], bool]:
        projection_notes: list[str] = []
        projected = False
        required_categories = list(getattr(qos_profile, "required_categories", []) or [])
        selected_categories = {
            canonical_xapp_category(xapp_name)
            for xapp_name in list(selected_xapps or [])
        }
        updated_selected = list(selected_xapps or [])
        for required_category in required_categories:
            if required_category in selected_categories:
                continue
            required_name = self._preferred_variant_name(
                required_category,
                available_xapp_names=available_xapp_names,
            )
            if required_name is not None:
                updated_selected.append(required_name)
                selected_categories.add(required_category)
                projection_notes.append(f"added_qos_guard:{required_category}")
                projected = True
        while len(updated_selected) > self.max_selected_xapps:
            before = list(updated_selected)
            updated_selected = self._remove_lowest_priority(
                updated_selected,
                goal_focus=goal_focus,
                risk_posture=risk_posture,
                protected_categories=set(required_categories),
            )
            removed = sorted(set(before) - set(updated_selected))
            if len(removed) > 0:
                projection_notes.append(f"qos_budget_prune:{removed[0]}")
            projected = True
        return updated_selected, projection_notes, projected

    def _build_chi_profile(self, profile_name: str) -> ArbitrationProfileRef:
        template = _CHI_PROFILE_LIBRARY.get(profile_name, _CHI_PROFILE_LIBRARY["chi:balanced_weighted"])
        return replace(template)

    def _default_eta_profile_name(self, *, goal_focus: str, risk_posture: str) -> str:
        if goal_focus == "mobility":
            return "eta:mobility_stable"
        if goal_focus in {"reliability", "throughput_reliability"} or risk_posture == "conservative":
            return "eta:reliability_guarded"
        if goal_focus == "throughput" and risk_posture != "conservative":
            return "eta:throughput_push"
        return "eta:balanced"

    def _default_chi_profile_name(self, *, goal_focus: str, risk_posture: str) -> str:
        if goal_focus in {"reliability", "throughput_reliability"} or risk_posture == "conservative":
            return "chi:guardrail_first"
        if goal_focus == "throughput" and risk_posture == "aggressive":
            return "chi:priority_scheduler"
        return "chi:balanced_weighted"

    def project_candidate(
        self,
        *,
        candidate_id: str,
        requested_xapps: list[str],
        goal,
        observation: Observation,
        available_xapp_names: list[str] | None = None,
        requested_eta_profile: str = "",
        requested_chi_profile: str = "",
    ) -> OrchestrationCandidatePlan:
        available_xapp_names = [
            str(item).strip()
            for item in list(available_xapp_names or requested_xapps or [])
            if str(item).strip() != ""
        ]
        requested_xapps = [
            str(item).strip()
            for item in list(requested_xapps or [])
            if str(item).strip() != ""
        ]
        selected_xapps: list[str] = []
        seen_categories: set[str] = set()
        projection_notes: list[str] = []
        projected = False
        goal_focus = self._goal_focus(goal)
        risk_posture = str(getattr(goal, "risk_posture", "balanced") or "balanced").strip().lower()

        for xapp_name in requested_xapps:
            category_name = canonical_xapp_category(xapp_name)
            if category_name in seen_categories:
                continue
            preferred_name = self._preferred_variant_name(
                xapp_name,
                available_xapp_names=available_xapp_names,
            )
            if preferred_name is None:
                projection_notes.append(f"dropped_unavailable:{xapp_name}")
                projected = True
                continue
            selected_xapps.append(preferred_name)
            seen_categories.add(category_name)

        for dependent_category, dependency_category in self.dependency_category_pairs:
            selected_categories = {
                canonical_xapp_category(xapp_name)
                for xapp_name in list(selected_xapps or [])
            }
            if dependent_category not in selected_categories or dependency_category in selected_categories:
                continue
            dependency_name = self._preferred_variant_name(
                dependency_category,
                available_xapp_names=available_xapp_names,
            )
            if dependency_name is not None and len(selected_xapps) < max(self.max_selected_xapps + 1, len(selected_xapps) + 1):
                selected_xapps.append(dependency_name)
                projection_notes.append(f"added_dependency:{dependent_category}->{dependency_category}")
            else:
                selected_xapps = [
                    xapp_name
                    for xapp_name in list(selected_xapps or [])
                    if canonical_xapp_category(xapp_name) != dependent_category
                ]
                projection_notes.append(f"dropped_for_missing_dependency:{dependent_category}->{dependency_category}")
            projected = True

        for left_category, right_category in self.mutex_category_pairs:
            selected_categories = {
                canonical_xapp_category(xapp_name): xapp_name
                for xapp_name in list(selected_xapps or [])
            }
            if left_category not in selected_categories or right_category not in selected_categories:
                continue
            left_priority = self._category_priority(
                left_category,
                goal_focus=goal_focus,
                risk_posture=risk_posture,
            )
            right_priority = self._category_priority(
                right_category,
                goal_focus=goal_focus,
                risk_posture=risk_posture,
            )
            drop_category = right_category if left_priority >= right_priority else left_category
            selected_xapps = [
                xapp_name
                for xapp_name in list(selected_xapps or [])
                if canonical_xapp_category(xapp_name) != drop_category
            ]
            projection_notes.append(f"resolved_mutex:{left_category}!={right_category}:dropped={drop_category}")
            projected = True

        while len(selected_xapps) > self.max_selected_xapps:
            before = list(selected_xapps)
            selected_xapps = self._remove_lowest_priority(
                selected_xapps,
                goal_focus=goal_focus,
                risk_posture=risk_posture,
            )
            removed = sorted(set(before) - set(selected_xapps))
            if len(removed) > 0:
                projection_notes.append(f"pruned_for_cardinality:{removed[0]}")
            projected = True

        total_cpu_cost, total_mem_cost = self._summed_costs(selected_xapps)
        while selected_xapps and (
            float(total_cpu_cost) > self.max_cpu_cost or float(total_mem_cost) > self.max_mem_cost
        ):
            before = list(selected_xapps)
            selected_xapps = self._remove_lowest_priority(
                selected_xapps,
                goal_focus=goal_focus,
                risk_posture=risk_posture,
            )
            total_cpu_cost, total_mem_cost = self._summed_costs(selected_xapps)
            removed = sorted(set(before) - set(selected_xapps))
            if len(removed) > 0:
                projection_notes.append(
                    f"pruned_for_budget:{removed[0]}:cpu={round(total_cpu_cost, 3)}:mem={round(total_mem_cost, 3)}"
                )
            projected = True

        qos_protection_profile = self._derive_qos_protection_profile(
            goal=goal,
            observation=observation,
            selected_xapps=selected_xapps,
        )
        selected_xapps, qos_projection_notes, qos_projected = self._qos_admissibility_projection(
            selected_xapps=selected_xapps,
            qos_profile=qos_protection_profile,
            goal_focus=goal_focus,
            risk_posture=risk_posture,
            available_xapp_names=available_xapp_names,
        )
        if qos_projected:
            projection_notes.extend(qos_projection_notes)
            projected = True
            total_cpu_cost, total_mem_cost = self._summed_costs(selected_xapps)

        qos_required_categories = {
            str(category_name)
            for category_name in list(getattr(qos_protection_profile, "required_categories", []) or [])
            if str(category_name).strip() != ""
        }
        for _ in range(8):
            hard_closure_changed = False
            selected_categories = {
                canonical_xapp_category(xapp_name)
                for xapp_name in list(selected_xapps or [])
            }
            for dependent_category, dependency_category in self.dependency_category_pairs:
                if dependent_category not in selected_categories or dependency_category in selected_categories:
                    continue
                dependency_name = self._preferred_variant_name(
                    dependency_category,
                    available_xapp_names=available_xapp_names,
                )
                trial_selection = list(selected_xapps)
                if dependency_name is not None:
                    trial_selection.append(dependency_name)
                trial_cpu_cost, trial_mem_cost = self._summed_costs(trial_selection)
                if (
                    dependency_name is not None
                    and len(trial_selection) <= self.max_selected_xapps
                    and float(trial_cpu_cost) <= self.max_cpu_cost
                    and float(trial_mem_cost) <= self.max_mem_cost
                ):
                    selected_xapps.append(dependency_name)
                    projection_notes.append(f"final_added_dependency:{dependent_category}->{dependency_category}")
                else:
                    selected_xapps = [
                        xapp_name
                        for xapp_name in list(selected_xapps or [])
                        if canonical_xapp_category(xapp_name) != dependent_category
                    ]
                    projection_notes.append(f"final_dropped_for_missing_dependency:{dependent_category}->{dependency_category}")
                projected = True
                hard_closure_changed = True
                break
            if hard_closure_changed:
                continue

            for left_category, right_category in self.mutex_category_pairs:
                selected_category_map = {
                    canonical_xapp_category(xapp_name): xapp_name
                    for xapp_name in list(selected_xapps or [])
                }
                if left_category not in selected_category_map or right_category not in selected_category_map:
                    continue
                left_priority = self._category_priority(
                    left_category,
                    goal_focus=goal_focus,
                    risk_posture=risk_posture,
                )
                right_priority = self._category_priority(
                    right_category,
                    goal_focus=goal_focus,
                    risk_posture=risk_posture,
                )
                drop_category = right_category if left_priority >= right_priority else left_category
                selected_xapps = [
                    xapp_name
                    for xapp_name in list(selected_xapps or [])
                    if canonical_xapp_category(xapp_name) != drop_category
                ]
                projection_notes.append(f"final_resolved_mutex:{left_category}!={right_category}:dropped={drop_category}")
                projected = True
                hard_closure_changed = True
                break
            if hard_closure_changed:
                continue

            if len(selected_xapps) > self.max_selected_xapps:
                before = list(selected_xapps)
                selected_xapps = self._remove_lowest_priority(
                    selected_xapps,
                    goal_focus=goal_focus,
                    risk_posture=risk_posture,
                    protected_categories=qos_required_categories,
                )
                removed = sorted(set(before) - set(selected_xapps))
                if len(removed) > 0:
                    projection_notes.append(f"final_pruned_for_cardinality:{removed[0]}")
                projected = True
                hard_closure_changed = True
                continue

            total_cpu_cost, total_mem_cost = self._summed_costs(selected_xapps)
            if selected_xapps and (
                float(total_cpu_cost) > self.max_cpu_cost or float(total_mem_cost) > self.max_mem_cost
            ):
                before = list(selected_xapps)
                selected_xapps = self._remove_lowest_priority(
                    selected_xapps,
                    goal_focus=goal_focus,
                    risk_posture=risk_posture,
                    protected_categories=qos_required_categories,
                )
                total_cpu_cost, total_mem_cost = self._summed_costs(selected_xapps)
                removed = sorted(set(before) - set(selected_xapps))
                if len(removed) > 0:
                    projection_notes.append(
                        f"final_pruned_for_budget:{removed[0]}:cpu={round(total_cpu_cost, 3)}:mem={round(total_mem_cost, 3)}"
                    )
                projected = True
                hard_closure_changed = True
                continue

            if not hard_closure_changed:
                break

        if len(selected_xapps) == 0 and len(available_xapp_names) > 0:
            fallback_name = self._preferred_variant_name(
                "urllc_guard_xapp" if goal_focus == "reliability" else "massive_mimo_beamformer_xapp",
                available_xapp_names=available_xapp_names,
            )
            if fallback_name is not None:
                selected_xapps = [fallback_name]
                total_cpu_cost, total_mem_cost = self._summed_costs(selected_xapps)
                projection_notes.append("fallback_singleton_plan")
                projected = True

        eta_profile_name = str(requested_eta_profile or self._default_eta_profile_name(goal_focus=goal_focus, risk_posture=risk_posture))
        chi_profile_name = str(requested_chi_profile or self._default_chi_profile_name(goal_focus=goal_focus, risk_posture=risk_posture))
        feasibility_summary = FeasibilityConstraintSummary(
            feasible=len(selected_xapps) > 0,
            projected=bool(projected),
            max_selected_xapps=int(self.max_selected_xapps),
            max_cpu_cost=float(self.max_cpu_cost),
            max_mem_cost=float(self.max_mem_cost),
            total_cpu_cost=float(total_cpu_cost),
            total_mem_cost=float(total_mem_cost),
            dependency_pairs=[f"{left}->{right}" for left, right in self.dependency_category_pairs],
            mutex_pairs=[f"{left}!={right}" for left, right in self.mutex_category_pairs],
            qos_constraint_count=int(len(getattr(qos_protection_profile, "constraints", []) or [])),
            qos_projection_applied=bool(qos_projected),
            projection_notes=list(projection_notes),
        )
        return OrchestrationCandidatePlan(
            candidate_id=str(candidate_id or ""),
            requested_xapps=list(requested_xapps),
            x=list(selected_xapps),
            B=default_hard_safety_envelope(selected_xapps, goal),
            eta=self._build_eta_profile(
                eta_profile_name,
                goal=goal,
                selected_xapps=selected_xapps,
            ),
            chi=self._build_chi_profile(chi_profile_name),
            forbidden_combinations=default_forbidden_combinations(selected_xapps),
            exploration_health_gate=default_exploration_health_gate(goal),
            qos_protection_profile=qos_protection_profile,
            feasibility_summary=feasibility_summary,
        )
