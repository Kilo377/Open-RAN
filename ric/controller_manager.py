"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: controller_manager.py
Description: This file defines the controller manager used by the simulator.
The controller manager schedules multiple Non-RT rApps and Near-RT xApps,
updates the Non-RT policy store by policy epochs, and generates one merged
executable slot-level action from xApps only with field ownership,
priority rules, and xApp activation control.
"""

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple

from models.action import (
    CommandAction,
    ConfigAction,
    ControlArbitrationTrace,
    ControlDecision,
    ControlMetadata,
    DroppedControlItem,
    FieldArbitrationResult,
    PolicyConstraint,
)
from models.observation import Observation
from models.oran_lifecycle import (
    E2ControlRequest,
    E2Indication,
    E2Subscription,
    OranLifecycleResources,
)
from core.debug import debug_print, orchestration_print
from models.intent_orchestration import a1_guidance_profile_payload, qos_protection_profile_payload
from ric.interfaces.a1_like import A1LikeInterface
from ric.interfaces.e2_like import E2LikeInterface
from ric.non_rt.policy_store import PolicyStore


class ControllerManager:
    """
    Manages multiple Non-RT and Near-RT controllers and builds the final
    executable slot-level action.
    INPUT
    a1_interfaces: The A1-like interfaces connected to Non-RT rApps.
    e2_interfaces: The E2-like interfaces connected to Near-RT xApps.
    xapp_interval_slots: The xApp execution interval in slots.
    rapp_interval_slots: The rApp execution interval in slots.
    OUTPUT
    A controller manager object that returns one final executable action per slot.
    EXAMPLE OUTPUT
    manager = ControllerManager(a1_interfaces, e2_interfaces, 5, 10)
    final_action = manager.update(slot_id=10, observation=observation)
    """

    ACTION_FIELDS = [
        "prb_power_w", "clear_prb_power", "power_group_size_prbs", "ue_prb_limits",
        "neighbor_cell_ids", "cio_offsets_db", "ret_enabled", "ret_tilt_deg", "ret_codebook_id", "time_to_trigger_ms",
        "codebook_id",
        "tx_power_dbm",
        "beam_profile_id",
        "scheduler_mode",
        "scheduler_policy_hint",
        "latency_bias",
        "slice_weight_adjustment",
        "beam_candidate_preference",
        "power_cap_dbm",
        "cell_sleep_mode",
        "traffic_steering_preference",
        "mobility_bias_db",
        "handover_cost_db",
        "handover_interruption_slots",
        "steering_hysteresis_db",
        "measurement_event_trigger",
        "measurement_event_offset_db",
    ]

    def __init__(
        self,
        a1_interfaces: list[A1LikeInterface],
        e2_interfaces: list[E2LikeInterface],
        xapp_interval_slots: int,
        rapp_interval_slots: int,
        run_rapps_on_start: bool = False,
        orchestration_warmup_slots: int = 0,
        active_intent_manager=None,
        variant_name: str = "full_method",
        validator_enabled: bool = True,
        rl_guidance_enabled: bool = True,
        safety_shield_enabled: bool = True,
        static_xapp_selection_enabled: bool = False,
        reflection_enabled: bool = True,
        memory_mode: str = "full",
        selection_mode: str = "reflective",
        repair_mode: str = "deterministic",
        conflict_mode: str = "typed_taxonomy",
        llm_pipeline_mode: str = "schema_bound",
        near_rt_mode: str = "projected_tuning",
        non_rt_reselection_enabled: bool = False,
        non_rt_reselection_fallback_threshold: int = 2,
        non_rt_reselection_min_gap_slots: int = 10,
    ) -> None:
        """
        Initializes the controller manager.
        INPUT
        a1_interfaces: The A1-like interfaces connected to Non-RT rApps.
        e2_interfaces: The E2-like interfaces connected to Near-RT xApps.
        xapp_interval_slots: The xApp execution interval in slots.
        rapp_interval_slots: The rApp execution interval in slots.
        OUTPUT
        None.
        EXAMPLE OUTPUT
        self.xapp_interval_slots = 5
        self.rapp_interval_slots = 10
        """
        self.a1_interfaces = a1_interfaces
        self.e2_interfaces = e2_interfaces
        self.xapp_interval_slots = xapp_interval_slots
        self.rapp_interval_slots = rapp_interval_slots
        self.run_rapps_on_start = bool(run_rapps_on_start)
        self.orchestration_warmup_slots = max(0, int(orchestration_warmup_slots))
        self.active_intent_manager = active_intent_manager
        self.variant_name = str(variant_name or "full_method")
        self.validator_enabled = bool(validator_enabled)
        self.rl_guidance_enabled = bool(rl_guidance_enabled)
        self.safety_shield_enabled = bool(safety_shield_enabled)
        self.static_xapp_selection_enabled = bool(static_xapp_selection_enabled)
        self.reflection_enabled = bool(reflection_enabled)
        self.memory_mode = str(memory_mode or "full")
        self.selection_mode = str(selection_mode or "reflective")
        self.repair_mode = str(repair_mode or "deterministic")
        self.conflict_mode = str(conflict_mode or "typed_taxonomy")
        self.llm_pipeline_mode = str(llm_pipeline_mode or "schema_bound")
        self.near_rt_mode = str(near_rt_mode or "projected_tuning")
        self.non_rt_reselection_enabled = bool(non_rt_reselection_enabled)
        self.non_rt_reselection_fallback_threshold = max(1, int(non_rt_reselection_fallback_threshold or 1))
        self.non_rt_reselection_min_gap_slots = max(1, int(non_rt_reselection_min_gap_slots or 1))
        self._initial_rapp_run_completed = False
        self._last_emitted_selection_signature: tuple[str, tuple[str, ...]] | None = None
        self._near_rt_fallback_streak = 0
        self._last_forced_non_rt_reselection_slot = -10**9
        self._current_non_rt_reselection_reason = ""
        self.orchestration_epoch_history: list[dict[str, Any]] = []
        self.current_action = ConfigAction()
        self.current_action_by_scope: Dict[str, ConfigAction] = {
            "cell:__default__": ConfigAction(apply_scope="cell", target_cell_id=None)
        }
        self.policy_store = PolicyStore()
        from ric.near_rt.xapp_library.selection import InstanceSelection
        self.instance_selection=InstanceSelection(self)
        if self.instance_selection.apps():self.policy_store.binding_validator=self.instance_selection.validate
        self.policy_store.derive_rl_guidance_when_missing = bool(self.rl_guidance_enabled)
        self._decision_seq = 0
        self._e2_resource_seq = 0
        self.e2_subscriptions_by_id: Dict[str, E2Subscription] = {}
        self.e2_subscription_id_by_subscriber: Dict[str, str] = {}
        self.e2_indications_by_id: Dict[str, E2Indication] = {}
        self.e2_control_requests_by_id: Dict[str, E2ControlRequest] = {}
        self._register_default_e2_subscriptions()

    def _to_serializable(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            return self._to_serializable(asdict(value))
        if isinstance(value, dict):
            return {str(key): self._to_serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _emit_rapp_summary(self, slot_id: int, policy_update) -> None:
        if getattr(policy_update, "active_intent", None) is not None:
            active_intent = getattr(policy_update, "active_intent", None)
            active_intent_summary = {
                "active_intent_id": str(
                    getattr(policy_update, "active_intent_id", "")
                    or getattr(active_intent, "intent_id", "")
                    or ""
                ),
                "intent_changed_this_epoch": bool(
                    getattr(policy_update, "intent_changed_this_epoch", False)
                ),
                "translation_refreshed_this_epoch": bool(
                    getattr(policy_update, "translation_refreshed_this_epoch", False)
                ),
                "last_update_epoch": getattr(active_intent, "last_update_epoch", None),
                "last_change_reason": str(getattr(active_intent, "last_change_reason", "") or ""),
            }
            orchestration_print(
                f"[active_intent][slot={slot_id:03d}] "
                f"{json.dumps(active_intent_summary, ensure_ascii=False, sort_keys=True)}"
            )
        if getattr(policy_update, "structured_goal", None) is not None:
            goal_summary = self._to_serializable(policy_update.structured_goal)
            if isinstance(goal_summary, dict):
                goal_summary["active_intent_id"] = str(
                    getattr(policy_update, "active_intent_id", "")
                    or getattr(getattr(policy_update, "structured_goal", None), "intent_id", "")
                    or ""
                )
                goal_summary["intent_changed_this_epoch"] = bool(
                    getattr(policy_update, "intent_changed_this_epoch", False)
                )
                goal_summary["translation_refreshed_this_epoch"] = bool(
                    getattr(policy_update, "translation_refreshed_this_epoch", False)
                )
            orchestration_print(
                f"[rapp_goal][slot={slot_id:03d}] "
                f"{json.dumps(goal_summary, ensure_ascii=False, sort_keys=True)}"
            )
        if getattr(policy_update, "validated_intent_template", None) is not None:
            validated_template = getattr(policy_update, "validated_intent_template", None)
            validation_report = getattr(validated_template, "validation_report", None)
            projection_report = getattr(validated_template, "projection_report", None)
            fallback_report = getattr(validated_template, "fallback_report", None)
            raw_candidate = dict(getattr(validated_template, "raw_llm_candidate", {}) or {})
            raw_kpi_targets = raw_candidate.get("kpi_targets", [])
            translation_query_key = getattr(validated_template, "translation_query_key", None)
            retrieved_context_summary = dict(
                getattr(validated_template, "retrieved_context_summary", {}) or {}
            )
            retrieved_context_traces = list(
                getattr(validated_template, "retrieved_context_traces", []) or []
            )
            validation_summary = {
                "template_id": str(getattr(validated_template, "template_id", "") or ""),
                "goal_id": str(getattr(validated_template, "goal_id", "") or ""),
                "active_intent_id": str(getattr(policy_update, "active_intent_id", "") or ""),
                "intent_changed_this_epoch": bool(
                    getattr(policy_update, "intent_changed_this_epoch", False)
                ),
                "translation_refreshed_this_epoch": bool(
                    getattr(policy_update, "translation_refreshed_this_epoch", False)
                ),
                "schema_ok": getattr(validation_report, "schema_ok", True),
                "admissibility_ok": getattr(validation_report, "admissibility_ok", True),
                "direction_ok": getattr(validation_report, "direction_ok", True),
                "range_ok": getattr(validation_report, "range_ok", True),
                "duplicate_kpis": list(getattr(validation_report, "duplicate_kpis", []) or []),
                "repaired_fields": list(getattr(validation_report, "repaired_fields", []) or [])
                + list(getattr(projection_report, "repaired_fields", []) or []),
                "rejected_kpis": list(getattr(validation_report, "rejected_kpis", []) or []),
                "projected": bool(getattr(projection_report, "projected", False)),
                "fallback_used": bool(getattr(fallback_report, "fallback_used", False)),
                "fallback_reason": str(getattr(fallback_report, "fallback_reason", "") or ""),
                "raw_kpi_count": len(raw_kpi_targets) if isinstance(raw_kpi_targets, list) else 0,
                "final_kpi_count": int(len(getattr(validated_template, "kpi_template", []) or [])),
                "translation_query_id": str(getattr(translation_query_key, "query_id", "") or ""),
                "translation_query_encoding_version": str(
                    getattr(translation_query_key, "encoding_version", "") or ""
                ),
                "translation_query_embedding_dim": int(
                    len(getattr(translation_query_key, "embedding_vector", []) or [])
                ),
                "translation_retrieved_context_count": int(len(retrieved_context_traces)),
                "translation_retrieved_context_source_ids": [
                    str(getattr(trace, "source_id", "") or "")
                    for trace in retrieved_context_traces[:5]
                    if str(getattr(trace, "source_id", "") or "").strip() != ""
                ],
                "translation_retrieved_context_summary": self._to_serializable(
                    retrieved_context_summary
                ),
            }
            orchestration_print(
                f"[rapp_goal_validation][slot={slot_id:03d}] "
                f"{json.dumps(validation_summary, ensure_ascii=False, sort_keys=True)}"
            )
        if getattr(policy_update, "xapp_selection_plan", None) is not None:
            plan = policy_update.xapp_selection_plan
            plan_view = getattr(policy_update, "orchestration_plan_view", None)
            if plan_view is None:
                plan_view = getattr(self.policy_store.get_snapshot(), "orchestration_plan_view", None)
            arbitration_profile = getattr(plan_view, "chi", None) if plan_view is not None else None
            memory_state = getattr(policy_update, "memory_state", None)
            selection_signature = (
                str(getattr(plan, "plan_id", "") or ""),
                tuple(str(item) for item in list(getattr(plan, "selected_xapps", []) or [])),
            )
            if selection_signature != self._last_emitted_selection_signature:
                self._last_emitted_selection_signature = selection_signature
                selection_summary = {
                    "plan_id": getattr(plan, "plan_id", ""),
                    "goal_id": getattr(plan, "goal_id", ""),
                    "selected_xapps": list(getattr(plan, "selected_xapps", []) or []),
                    "selection_query_key": self._to_serializable(
                        getattr(plan, "selection_query_key", None)
                    ),
                    "top_candidate_plan_scores": self._to_serializable(
                        list(getattr(plan, "candidate_plan_scores", []) or [])[:3]
                    ),
                    "top_retrieved_success_traces": self._to_serializable(
                        list(getattr(memory_state, "last_retrieved_success_traces", []) or [])[:3]
                    ),
                    "top_retrieved_failure_traces": self._to_serializable(
                        list(getattr(memory_state, "last_retrieved_failure_traces", []) or [])[:3]
                    ),
                    "top_candidate_plan_projections": self._to_serializable(
                        list(getattr(plan, "candidate_plan_projections", []) or [])[:3]
                    ),
                    "last_failure_reason": getattr(plan, "last_failure_reason", ""),
                    "last_failure_gap_score": getattr(plan, "last_failure_gap_score", None),
                    "eta": self._to_serializable(
                        a1_guidance_profile_payload(getattr(plan, "eta", None))
                    ),
                    "hard_safety_envelope": self._to_serializable(
                        list(getattr(plan, "hard_safety_envelope", []) or [])
                    ),
                    "forbidden_combinations": self._to_serializable(
                        list(getattr(plan, "forbidden_combinations", []) or [])
                    ),
                    "exploration_health_gate": self._to_serializable(
                        getattr(plan, "exploration_health_gate", None)
                    ),
                    "qos_protection_profile": self._to_serializable(
                        qos_protection_profile_payload(
                            getattr(plan, "qos_protection_profile", None)
                        )
                    ),
                    "composition_profile": {
                        "profile_name": str(getattr(arbitration_profile, "profile_name", "") or ""),
                        "composition_mode": str(getattr(arbitration_profile, "composition_mode", "") or ""),
                        "arbitration_mode": str(getattr(arbitration_profile, "arbitration_mode", "") or ""),
                        "override_order": list(getattr(arbitration_profile, "override_order", []) or []),
                        "defaulted": bool(getattr(arbitration_profile, "defaulted", True)),
                    }
                    if arbitration_profile is not None
                    else None,
                    "feasibility_summary": self._to_serializable(
                        getattr(plan, "feasibility_summary", None)
                    ),
                }
                orchestration_print(
                    f"[rapp_selection][slot={slot_id:03d}] "
                    f"{json.dumps(selection_summary, ensure_ascii=False, sort_keys=True)}"
                )
                if getattr(plan, "reflection", None) is not None:
                    reflection_summary = {
                        "plan_id": getattr(plan, "plan_id", ""),
                        "goal_id": getattr(plan, "goal_id", ""),
                        "failure_reason": getattr(plan, "last_failure_reason", ""),
                        "failure_gap_score": getattr(plan, "last_failure_gap_score", None),
                        "reflection": self._to_serializable(getattr(plan, "reflection", None)),
                    }
                    orchestration_print(
                        f"[rapp_reflection][slot={slot_id:03d}] "
                        f"{json.dumps(reflection_summary, ensure_ascii=False, sort_keys=True)}"
                    )
        if getattr(policy_update, "reward_spec", None) is not None:
            reward_spec = policy_update.reward_spec
            reward_summary = {
                "reward_id": getattr(reward_spec, "reward_id", ""),
                "goal_id": getattr(reward_spec, "goal_id", ""),
                "selection_plan_id": getattr(reward_spec, "selection_plan_id", ""),
                "reward_terms": dict(getattr(reward_spec, "reward_terms", {}) or {}),
                "kpi_weights": dict(getattr(reward_spec, "kpi_weights", {}) or {}),
                "black_box_xapps": list(getattr(reward_spec, "black_box_xapps", []) or []),
            }
            orchestration_print(
                f"[rapp_reward][slot={slot_id:03d}] "
                f"{json.dumps(reward_summary, ensure_ascii=False, sort_keys=True)}"
            )
        if getattr(policy_update, "rl_guidance", None) is not None:
            rl_guidance = policy_update.rl_guidance
            selected_safety_rule_set = getattr(rl_guidance, "selected_safety_rule_set", None)
            guidance_summary = {
                "guidance_id": getattr(rl_guidance, "guidance_id", ""),
                "goal_id": getattr(rl_guidance, "goal_id", ""),
                "selection_plan_id": getattr(rl_guidance, "selection_plan_id", ""),
                "source_epoch": getattr(rl_guidance, "source_epoch", None),
                "derived_from_intent_id": getattr(rl_guidance, "derived_from_intent_id", ""),
                "fallback_used": bool(getattr(rl_guidance, "fallback_used", False)),
                "guided_action_region_count": int(
                    len(list(getattr(rl_guidance, "guided_action_region", []) or []))
                ),
                "parameter_priority_vector": dict(
                    getattr(rl_guidance, "parameter_priority_vector", {}) or {}
                ),
                "selected_safety_rule_set": {
                    "rule_set_id": getattr(selected_safety_rule_set, "rule_set_id", ""),
                    "selected_rule_ids": [
                        str(getattr(rule_ref, "rule_id", "") or "")
                        for rule_ref in list(
                            getattr(selected_safety_rule_set, "selected_rule_refs", []) or []
                        )
                    ],
                }
                if selected_safety_rule_set is not None
                else None,
            }
            orchestration_print(
                f"[rapp_guidance][slot={slot_id:03d}] "
                f"{json.dumps(guidance_summary, ensure_ascii=False, sort_keys=True)}"
            )

    def _current_variant_flags(self) -> dict[str, object]:
        return {
            "variant_name": str(self.variant_name or "full_method"),
            "validator_enabled": bool(self.validator_enabled),
            "rl_guidance_enabled": bool(self.rl_guidance_enabled),
            "safety_shield_enabled": bool(self.safety_shield_enabled),
            "static_xapp_selection_enabled": bool(self.static_xapp_selection_enabled),
            "reflection_enabled": bool(self.reflection_enabled),
            "memory_mode": str(self.memory_mode),
            "selection_mode": str(self.selection_mode),
            "repair_mode": str(self.repair_mode),
            "conflict_mode": str(self.conflict_mode),
            "llm_pipeline_mode": str(self.llm_pipeline_mode),
            "near_rt_mode": str(self.near_rt_mode),
            "non_rt_reselection_enabled": bool(self.non_rt_reselection_enabled),
        }

    def _build_orchestration_epoch_entry(
        self,
        *,
        slot_id: int,
        observation: Observation,
    ) -> dict[str, Any]:
        from experiments.eval_metrics import compute_intent_loss

        policy_snapshot = self.policy_store.get_snapshot()
        structured_goal = getattr(policy_snapshot, "structured_goal", None)
        validated_template = getattr(policy_snapshot, "validated_intent_template", None)
        plan = getattr(policy_snapshot, "orchestration_plan_view", None)
        if plan is None:
            plan = getattr(policy_snapshot, "xapp_selection_plan", None)
        rl_guidance = getattr(policy_snapshot, "rl_guidance", None)
        safety_constraint_set = getattr(policy_snapshot, "safety_constraint_set", None)
        memory_state = getattr(policy_snapshot, "memory_state", None)
        arbitration_profile = getattr(plan, "chi", None) if plan is not None else None
        translation_query_key = (
            getattr(validated_template, "translation_query_key", None)
            if validated_template is not None
            else None
        )
        translation_context_summary = (
            dict(getattr(validated_template, "retrieved_context_summary", {}) or {})
            if validated_template is not None
            else {}
        )
        translation_context_traces = (
            list(getattr(validated_template, "retrieved_context_traces", []) or [])
            if validated_template is not None
            else []
        )
        plan_eta = {}
        if plan is not None:
            plan_eta = a1_guidance_profile_payload(getattr(plan, "eta", None))
        if len(plan_eta) == 0:
            plan_eta = a1_guidance_profile_payload(
                getattr(getattr(policy_snapshot, "xapp_selection_plan", None), "eta", None)
            )
        feasibility_summary = None
        if plan is not None:
            feasibility_summary = getattr(plan, "feasibility_summary", None)
        if feasibility_summary is None:
            feasibility_summary = getattr(
                getattr(policy_snapshot, "xapp_selection_plan", None),
                "feasibility_summary",
                None,
            )
        if safety_constraint_set is None and rl_guidance is not None:
            safety_constraint_set = getattr(rl_guidance, "selected_safety_rule_set", None)

        intent_loss = compute_intent_loss(
            observation=observation,
            validated_intent_template=validated_template,
            structured_goal=structured_goal,
        )
        selected_xapps = []
        if plan is not None:
            selected_xapps = list(
                getattr(plan, "x", getattr(plan, "selected_xapps", [])) or []
            )
        selection_query_key = getattr(
            getattr(policy_snapshot, "xapp_selection_plan", None),
            "selection_query_key",
            None,
        )
        selection_query_signature = dict(
            getattr(selection_query_key, "context_signature", {}) or {}
        )
        candidate_plan_scores = list(
            getattr(getattr(policy_snapshot, "xapp_selection_plan", None), "candidate_plan_scores", []) or []
        )
        selected_candidate_score = next(
            (
                candidate_score
                for candidate_score in candidate_plan_scores
                if bool(getattr(candidate_score, "selected", False))
            ),
            None,
        )

        health_gate = None
        if safety_constraint_set is not None:
            health_gate = getattr(safety_constraint_set, "exploration_health_gate", None)
        if health_gate is None and plan is not None:
            health_gate = getattr(plan, "exploration_health_gate", None)

        selected_rule_ids: list[str] = []
        if safety_constraint_set is not None:
            selected_rule_ids = [
                str(getattr(rule_ref, "rule_id", "") or "")
                for rule_ref in list(getattr(safety_constraint_set, "selected_rule_refs", []) or [])
                if str(getattr(rule_ref, "rule_id", "") or "").strip() != ""
            ]
        retrieved_success_traces = list(
            getattr(memory_state, "last_retrieved_success_traces", []) or []
        )
        retrieved_failure_traces = list(
            getattr(memory_state, "last_retrieved_failure_traces", []) or []
        )

        hard_safety_envelope = []
        forbidden_combinations = []
        guided_action_region = []
        if safety_constraint_set is not None:
            hard_safety_envelope = list(
                getattr(safety_constraint_set, "safety_envelope", []) or []
            )
            forbidden_combinations = list(
                getattr(safety_constraint_set, "forbidden_combinations", []) or []
            )
        if len(hard_safety_envelope) == 0 and plan is not None:
            hard_safety_envelope = list(
                getattr(plan, "B", getattr(plan, "hard_safety_envelope", [])) or []
            )
            forbidden_combinations = list(
                getattr(plan, "forbidden_combinations", []) or []
            )
        if rl_guidance is not None:
            guided_action_region = list(getattr(rl_guidance, "guided_action_region", []) or [])

        entry = {
            "epoch_id": int(getattr(policy_snapshot, "policy_epoch_id", 0) or 0),
            "slot_id": int(slot_id),
            "active_intent_id": str(getattr(policy_snapshot, "active_intent_id", "") or ""),
            "intent_changed_this_epoch": bool(
                getattr(policy_snapshot, "intent_changed_this_epoch", False)
            ),
            "translation_refreshed_this_epoch": bool(
                getattr(policy_snapshot, "translation_refreshed_this_epoch", False)
            ),
            "translation_query_id": str(getattr(translation_query_key, "query_id", "") or ""),
            "translation_query_encoding_version": str(
                getattr(translation_query_key, "encoding_version", "") or ""
            ),
            "translation_query_retriever_backend_name": str(
                getattr(translation_query_key, "retriever_backend_name", "") or ""
            ),
            "translation_query_embedding_dim": int(
                len(getattr(translation_query_key, "embedding_vector", []) or [])
            ),
            "translation_context_signature": dict(
                getattr(translation_query_key, "context_signature", {}) or {}
            ),
            "translation_retrieved_context_count": int(len(translation_context_traces)),
            "translation_retrieved_context_source_ids": [
                str(getattr(trace, "source_id", "") or "")
                for trace in translation_context_traces
                if str(getattr(trace, "source_id", "") or "").strip() != ""
            ],
            "translation_retrieved_context_summary": self._to_serializable(
                translation_context_summary
            ),
            "selected_xapps": [str(item) for item in selected_xapps],
            "selection_query_id": str(getattr(selection_query_key, "query_id", "") or ""),
            "selection_query_slot_id": getattr(selection_query_key, "query_slot_id", None),
            "selection_query_encoding_version": str(
                getattr(selection_query_key, "encoding_version", "") or ""
            ),
            "selection_query_retriever_backend_name": str(
                getattr(selection_query_key, "retriever_backend_name", "") or ""
            ),
            "selection_query_embedding_dim": int(
                len(getattr(selection_query_key, "embedding_vector", []) or [])
            ),
            "selection_query_kpis": list(getattr(selection_query_key, "kpi_names", []) or []),
            "selection_query_target_profile": str(
                selection_query_signature.get("target_profile", "") or ""
            ),
            "selection_query_previous_xapp_categories": list(
                getattr(selection_query_key, "previous_xapp_categories", []) or []
            ),
            "guidance_id": str(getattr(rl_guidance, "guidance_id", "") or ""),
            "selection_plan_id": str(
                getattr(plan, "plan_id", getattr(getattr(policy_snapshot, "xapp_selection_plan", None), "plan_id", ""))
                or ""
            ),
            "selection_status": str(getattr(plan, "status", "") or ""),
            "selection_locked": bool(getattr(plan, "selection_locked", False)),
            "selection_generation": int(getattr(plan, "selection_generation", 0) or 0),
            "last_failure_reason": str(getattr(plan, "last_failure_reason", "") or ""),
            "last_failure_gap_score": getattr(plan, "last_failure_gap_score", None),
            "candidate_plan_count": int(len(candidate_plan_scores)),
            "selected_candidate_id": str(
                getattr(selected_candidate_score, "candidate_id", "") or ""
            ),
            "selected_candidate_score_version": str(
                getattr(selected_candidate_score, "score_version", "") or ""
            ),
            "selected_candidate_score_direction": str(
                getattr(selected_candidate_score, "score_direction", "") or ""
            ),
            "selected_candidate_predictor_name": str(
                getattr(selected_candidate_score, "predictor_name", "") or ""
            ),
            "selected_candidate_total_score": (
                float(getattr(selected_candidate_score, "total_score", 0.0) or 0.0)
                if selected_candidate_score is not None
                else None
            ),
            "selected_candidate_predicted_intent_loss": (
                float(getattr(selected_candidate_score, "predicted_intent_loss", 0.0) or 0.0)
                if selected_candidate_score is not None
                else None
            ),
            "selected_candidate_conflict_cost": (
                float(
                    getattr(selected_candidate_score, "predicted_conflict_cost", 0.0)
                    or 0.0
                )
                if selected_candidate_score is not None
                else None
            ),
            "selected_candidate_stability_risk": (
                float(getattr(selected_candidate_score, "stability_risk", 0.0) or 0.0)
                if selected_candidate_score is not None
                else None
            ),
            "selected_candidate_resource_pressure": (
                float(getattr(selected_candidate_score, "resource_pressure", 0.0) or 0.0)
                if selected_candidate_score is not None
                else None
            ),
            "selected_candidate_rationale": (
                str(getattr(selected_candidate_score, "rationale", "") or "")
                if selected_candidate_score is not None
                else ""
            ),
            "selected_candidate_eta_profile_name": (
                str(getattr(selected_candidate_score, "eta_profile_name", "") or "")
                if selected_candidate_score is not None
                else ""
            ),
            "selected_candidate_chi_profile_name": (
                str(getattr(selected_candidate_score, "chi_profile_name", "") or "")
                if selected_candidate_score is not None
                else ""
            ),
            "selected_candidate_parameter_bound_count": (
                int(getattr(selected_candidate_score, "parameter_bound_count", 0) or 0)
                if selected_candidate_score is not None
                else 0
            ),
            "selected_candidate_feasible": (
                bool(getattr(selected_candidate_score, "feasible", False))
                if selected_candidate_score is not None
                else False
            ),
            "selected_candidate_projection_applied": (
                bool(getattr(selected_candidate_score, "projection_applied", False))
                if selected_candidate_score is not None
                else False
            ),
            "selected_candidate_total_cpu_cost": (
                float(getattr(selected_candidate_score, "total_cpu_cost", 0.0) or 0.0)
                if selected_candidate_score is not None
                else None
            ),
            "selected_candidate_total_mem_cost": (
                float(getattr(selected_candidate_score, "total_mem_cost", 0.0) or 0.0)
                if selected_candidate_score is not None
                else None
            ),
            "eta_profile_name": str(plan_eta.get("profile_name", "") or ""),
            "eta_goal_focus": str(plan_eta.get("goal_focus", "") or ""),
            "eta_payload": self._to_serializable(plan_eta),
            "qos_profile_id": str(
                qos_protection_profile_payload(
                    getattr(plan, "qos_protection_profile", None)
                ).get("profile_id", "")
                or ""
            ),
            "qos_constraint_count": int(
                len(
                    qos_protection_profile_payload(
                        getattr(plan, "qos_protection_profile", None)
                    ).get("constraints", [])
                    or []
                )
            ),
            "composition_profile_name": str(getattr(arbitration_profile, "profile_name", "") or ""),
            "composition_mode": str(getattr(arbitration_profile, "composition_mode", "") or ""),
            "arbitration_mode": str(getattr(arbitration_profile, "arbitration_mode", "") or ""),
            "override_order": list(getattr(arbitration_profile, "override_order", []) or []),
            "chi_defaulted": bool(getattr(arbitration_profile, "defaulted", True)),
            "chi_notes": list(getattr(arbitration_profile, "notes", []) or []),
            "feasible": bool(getattr(feasibility_summary, "feasible", True))
            if feasibility_summary is not None
            else True,
            "projection_applied": bool(getattr(feasibility_summary, "projected", False))
            if feasibility_summary is not None
            else False,
            "total_cpu_cost": (
                float(getattr(feasibility_summary, "total_cpu_cost", 0.0) or 0.0)
                if feasibility_summary is not None
                else None
            ),
            "total_mem_cost": (
                float(getattr(feasibility_summary, "total_mem_cost", 0.0) or 0.0)
                if feasibility_summary is not None
                else None
            ),
            "max_cpu_cost": (
                float(getattr(feasibility_summary, "max_cpu_cost", 0.0) or 0.0)
                if getattr(feasibility_summary, "max_cpu_cost", None) is not None
                else None
            ),
            "max_mem_cost": (
                float(getattr(feasibility_summary, "max_mem_cost", 0.0) or 0.0)
                if getattr(feasibility_summary, "max_mem_cost", None) is not None
                else None
            ),
            "dependency_pairs": list(getattr(feasibility_summary, "dependency_pairs", []) or []),
            "mutex_pairs": list(getattr(feasibility_summary, "mutex_pairs", []) or []),
            "projection_notes": list(getattr(feasibility_summary, "projection_notes", []) or []),
            "memory_enabled": bool(getattr(memory_state, "memory_enabled", False)),
            "success_memory_count": int(len(list(getattr(memory_state, "success_cases", []) or []))),
            "failure_memory_count": int(len(list(getattr(memory_state, "failure_cases", []) or []))),
            "last_retrieved_success_case_ids": list(
                getattr(memory_state, "last_retrieved_success_case_ids", []) or []
            ),
            "last_retrieved_failure_case_ids": list(
                getattr(memory_state, "last_retrieved_failure_case_ids", []) or []
            ),
            "last_retrieved_success_traces": self._to_serializable(retrieved_success_traces),
            "last_retrieved_failure_traces": self._to_serializable(retrieved_failure_traces),
            "goal_id": str(getattr(structured_goal, "goal_id", "") or ""),
            "validator_enabled": bool(self.validator_enabled),
            "rl_guidance_enabled": bool(self.rl_guidance_enabled),
            "safety_shield_enabled": bool(self.safety_shield_enabled),
            "static_xapp_selection_enabled": bool(self.static_xapp_selection_enabled),
            "variant_name": str(self.variant_name or "full_method"),
            "memory_mode": str(self.memory_mode),
            "selection_mode": str(self.selection_mode),
            "repair_mode": str(self.repair_mode),
            "conflict_mode": str(self.conflict_mode),
            "llm_pipeline_mode": str(self.llm_pipeline_mode),
            "near_rt_mode": str(self.near_rt_mode),
            "validator_schema_ok": (
                getattr(getattr(validated_template, "validation_report", None), "schema_ok", None)
                if validated_template is not None
                else None
            ),
            "validator_admissibility_ok": (
                getattr(getattr(validated_template, "validation_report", None), "admissibility_ok", None)
                if validated_template is not None
                else None
            ),
            "validator_range_ok": (
                getattr(getattr(validated_template, "validation_report", None), "range_ok", None)
                if validated_template is not None
                else None
            ),
            "guidance_source_epoch": getattr(rl_guidance, "source_epoch", None),
            "guidance_fallback_used": bool(getattr(rl_guidance, "fallback_used", False)),
            "guided_action_region_count": int(len(guided_action_region)),
            "guided_action_region_fields": sorted(
                {
                    str(getattr(parameter_range, "field_name", "") or "")
                    for parameter_range in guided_action_region
                    if str(getattr(parameter_range, "field_name", "") or "").strip() != ""
                }
            ),
            "hard_safety_envelope_count": int(len(hard_safety_envelope)),
            "hard_safety_envelope_fields": sorted(
                {
                    str(getattr(parameter_range, "field_name", "") or "")
                    for parameter_range in hard_safety_envelope
                    if str(getattr(parameter_range, "field_name", "") or "").strip() != ""
                }
            ),
            "forbidden_combination_count": int(len(forbidden_combinations)),
            "selected_rule_ids": list(selected_rule_ids),
            "exploration_health_gate": self._to_serializable(health_gate),
            "observable_snapshot": {
                "slot_id": int(getattr(observation, "slot_id", slot_id)),
                "total_queue_bytes": int(getattr(observation, "total_queue_bytes", 0) or 0),
                "total_throughput_mbps": float(
                    getattr(observation, "total_throughput_mbps", 0.0) or 0.0
                ),
                "avg_bler": (
                    float(getattr(observation, "avg_bler", 0.0))
                    if getattr(observation, "avg_bler", None) is not None
                    else None
                ),
                "handover_pending_ue_count": int(
                    getattr(observation, "handover_pending_ue_count", 0) or 0
                ),
            },
            "forced_non_rt_reselection": bool(self._current_non_rt_reselection_reason),
            "forced_non_rt_reselection_reason": str(self._current_non_rt_reselection_reason or ""),
        }
        entry.update(intent_loss)
        entry.update(self._current_variant_flags())
        return self._to_serializable(entry)

    def _record_orchestration_epoch(
        self,
        *,
        slot_id: int,
        observation: Observation,
    ) -> None:
        self.orchestration_epoch_history.append(
            self._build_orchestration_epoch_entry(
                slot_id=int(slot_id),
                observation=observation,
            )
        )

    def get_orchestration_epoch_history(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in list(self.orchestration_epoch_history)]

    def _next_decision_id(self, prefix: str) -> str:
        self._decision_seq += 1
        return f"{prefix}:{self._decision_seq}"

    def _next_e2_resource_id(self, prefix: str) -> str:
        self._e2_resource_seq += 1
        return f"{prefix}:{self._e2_resource_seq}"

    def _register_default_e2_subscriptions(self) -> None:
        for e2_interface in self.e2_interfaces:
            subscriber_name = str(getattr(getattr(e2_interface, "xapp", None), "name", "unknown_xapp"))
            if subscriber_name in self.e2_subscription_id_by_subscriber:
                continue
            subscription = E2Subscription(
                subscription_id=self._next_e2_resource_id("e2_subscription"),
                subscriber_name=subscriber_name,
                scope="cluster",
                metric_filter="full_observation",
                status="active",
            )
            self.e2_subscriptions_by_id[subscription.subscription_id] = subscription
            self.e2_subscription_id_by_subscriber[subscriber_name] = subscription.subscription_id

    def _summarize_observation(self, observation: Observation) -> dict[str, object]:
        return {
            "slot_id": int(getattr(observation, "slot_id", -1)),
            "cell_id": str(getattr(observation, "cell_id", "cell_0")),
            "runtime_mode": str(getattr(observation, "runtime_mode", "single_active_cell")),
            "total_queue_bytes": int(getattr(observation, "total_queue_bytes", 0)),
            "num_scheduled_ues": int(getattr(observation, "num_scheduled_ues", 0)),
            "avg_sinr_db": getattr(observation, "avg_sinr_db", None),
            "measurement_age_slots": getattr(observation, "measurement_age_slots", None),
            "cell_observation_count": int(len(getattr(observation, "cell_observations", []) or [])),
        }

    def _record_e2_indication(
        self,
        *,
        subscriber_name: str,
        observation: Observation,
        slot_id: int,
    ) -> None:
        subscription_id = self.e2_subscription_id_by_subscriber.get(str(subscriber_name))
        if subscription_id is None:
            return
        subscription = self.e2_subscriptions_by_id.get(subscription_id)
        if subscription is not None:
            subscription.updated_at_slot = int(slot_id)
        indication = E2Indication(
            indication_id=self._next_e2_resource_id("e2_indication"),
            subscription_id=subscription_id,
            producer_name="controller_manager",
            slot_id=int(slot_id),
            produced_at_slot=getattr(observation, "produced_at_slot", None),
            delivered_at_slot=getattr(observation, "delivered_at_slot", None),
            payload_summary=self._summarize_observation(observation),
        )
        self.e2_indications_by_id[indication.indication_id] = indication

    def _build_config_payload_summary(self, action: ConfigAction) -> dict[str, object]:
        field_names = [
            str(field_name)
            for field_name in self.ACTION_FIELDS
            if self._action_field_has_value(field_name, getattr(action, field_name))
        ]
        return {
            "apply_scope": str(getattr(action, "apply_scope", "cell")),
            "target_cell_id": getattr(action, "target_cell_id", None),
            "field_names": sorted(field_names),
        }

    def _summarize_config_action_for_debug(self, action: ConfigAction) -> str:
        payload = self._build_config_payload_summary(action)
        raw_field_names = payload.get("field_names", [])
        field_names = list(raw_field_names) if isinstance(raw_field_names, list) else []
        field_pairs: list[str] = []
        for field_name in field_names:
            field_value = getattr(action, str(field_name))
            field_pairs.append(f"{field_name}={field_value}")
        apply_scope = str(payload.get("apply_scope", "cell"))
        target_cell_id = payload.get("target_cell_id", None)
        target_text = f",target={target_cell_id}" if target_cell_id is not None else ""
        fields_text = ", ".join(field_pairs) if len(field_pairs) > 0 else "no_fields"
        return f"scope={apply_scope}{target_text} {{{fields_text}}}"

    def _summarize_config_actions_for_debug(self, actions: list[ConfigAction]) -> str:
        if len(actions) == 0:
            return "[]"
        return "[" + "; ".join(
            self._summarize_config_action_for_debug(action)
            for action in actions
        ) + "]"

    def _record_e2_control_request_for_config(
        self,
        *,
        action: ConfigAction,
        source_name: str,
        slot_id: int,
    ) -> None:
        field_names = [
            str(field_name)
            for field_name in self.ACTION_FIELDS
            if self._action_field_has_value(field_name, getattr(action, field_name))
        ]
        if len(field_names) == 0:
            return

        request_id = str(getattr(getattr(action, "metadata", None), "decision_id", "")).strip()
        if request_id == "":
            request_id = self._next_e2_resource_id("e2_control_request")
        action.control_request_id = request_id
        request = E2ControlRequest(
            request_id=request_id,
            source_name=str(source_name),
            kind="config",
            target_cell_id=getattr(action, "target_cell_id", None),
            target_cell_ids=(
                [str(getattr(action, "target_cell_id", None))]
                if getattr(action, "target_cell_id", None) is not None
                else []
            ),
            issued_at_slot=int(slot_id),
            updated_at_slot=int(slot_id),
            status="issued",
            reason="merged_config_ready_for_actuation",
            payload_summary=self._build_config_payload_summary(action),
        )
        self.e2_control_requests_by_id[request.request_id] = request

    def _record_e2_control_request_for_command(
        self,
        *,
        command: CommandAction,
        source_name: str,
        slot_id: int,
    ) -> None:
        request_id = str(getattr(getattr(command, "metadata", None), "decision_id", "")).strip()
        if request_id == "":
            request_id = self._next_e2_resource_id("e2_control_request")
        command.control_request_id = request_id
        request = E2ControlRequest(
            request_id=request_id,
            source_name=str(source_name),
            kind="command",
            issued_at_slot=int(slot_id),
            updated_at_slot=int(slot_id),
            status="issued",
            reason="command_ready_for_actuation",
            payload_summary={
                "command_type": str(getattr(command, "command_type", "")),
                "target_ue_id": getattr(command, "target_ue_id", None),
                "target_cell_id": getattr(command, "target_cell_id", None),
                "delta_priority": float(getattr(command, "delta_priority", 0.0)),
            },
        )
        self.e2_control_requests_by_id[request.request_id] = request

    def _build_oran_lifecycle_resources(self) -> OranLifecycleResources:
        a1_resources = self.policy_store.get_lifecycle_resources()
        return OranLifecycleResources(
            a1_policy_types=list(a1_resources.a1_policy_types),
            a1_policy_instances=list(a1_resources.a1_policy_instances),
            a1_policy_statuses=list(a1_resources.a1_policy_statuses),
            a1_ei_types=list(a1_resources.a1_ei_types),
            a1_ei_jobs=list(a1_resources.a1_ei_jobs),
            e2_subscriptions=list(self.e2_subscriptions_by_id.values()),
            e2_indications=list(self.e2_indications_by_id.values()),
            e2_control_requests=list(self.e2_control_requests_by_id.values()),
        )

    def _stamp_metadata(
        self,
        item: object,
        *,
        source: str,
        slot_id: int,
        priority: int | None = None,
        reason: str = "",
        decision_id_prefix: str = "control",
    ) -> None:
        metadata = getattr(item, "metadata", None)
        if metadata is None:
            return

        if getattr(metadata, "source", "unknown") == "unknown":
            metadata.source = source
        if getattr(metadata, "issued_at_slot", None) is None:
            metadata.issued_at_slot = int(slot_id)
        if getattr(metadata, "reason", "") == "":
            metadata.reason = reason
        if getattr(metadata, "priority", None) is None and priority is not None:
            metadata.priority = int(priority)
        if getattr(metadata, "decision_id", None) is None:
            metadata.decision_id = self._next_decision_id(decision_id_prefix)

        if hasattr(item, "source_name") and getattr(item, "source_name") == "unknown":
            try:
                setattr(item, "source_name", str(metadata.source))
            except Exception:
                pass

    def _action_field_has_value(self, field_name: str, field_value: object) -> bool:
        if field_value is None:
            return False
        if field_name == "slice_weight_adjustment" and isinstance(field_value, dict):
            return len(field_value) > 0
        if isinstance(field_value, str):
            return field_value.strip() != ""
        return True

    def _copy_action_field_value(self, field_value: object) -> object:
        if isinstance(field_value, dict):
            return dict(field_value)
        if isinstance(field_value, list):
            return list(field_value)
        return field_value

    def _normalize_current_action_surface(self) -> None:
        scheduler_hint = getattr(self.current_action, "scheduler_policy_hint", None)
        if scheduler_hint != "slice_bias":
            self.current_action.slice_weight_adjustment = {}
        if scheduler_hint != "latency_bias":
            self.current_action.latency_bias = None

    def _normalize_scope_key(
        self,
        action: ConfigAction,
    ) -> tuple[str, str, str | None]:
        apply_scope = str(getattr(action, "apply_scope", "cell")).strip().lower() or "cell"
        target_cell_id = getattr(action, "target_cell_id", None)
        normalized_target_cell_id = (
            str(target_cell_id).strip()
            if target_cell_id is not None and str(target_cell_id).strip() != ""
            else None
        )
        if apply_scope == "cluster":
            return ("cluster:*", "cluster", None)
        if normalized_target_cell_id is None:
            return ("cell:__default__", "cell", None)
        return (f"cell:{normalized_target_cell_id}", "cell", normalized_target_cell_id)

    def _build_scope_action(
        self,
        *,
        scope_key: str,
        apply_scope: str,
        target_cell_id: str | None,
    ) -> ConfigAction:
        action = self.current_action_by_scope.get(scope_key)
        if action is None:
            action = ConfigAction(
                apply_scope=str(apply_scope),
                target_cell_id=target_cell_id,
                source_name="controller_manager",
            )
            self.current_action_by_scope[scope_key] = action
        else:
            action.apply_scope = str(apply_scope)
            action.target_cell_id = target_cell_id
        return action

    def _run_rapps_if_needed(
        self,
        slot_id: int,
        observation: Observation,
        *,
        force: bool = False,
        force_reason: str = "",
    ) -> bool:
        """
        Runs all Non-RT rApps when the interval condition is satisfied and
        rebuilds the policy snapshot for the new rApp policy epoch.
        INPUT
        slot_id: The current simulation slot.
        observation: The current aggregated observation.
        OUTPUT
        None. The policy store may be replaced with a new epoch snapshot.
        EXAMPLE OUTPUT
        PolicyStore starts a new epoch and receives multiple PolicyUpdate objects.
        """
        first_allowed_slot = max(1, self.orchestration_warmup_slots + 1)
        if not self._initial_rapp_run_completed:
            should_run = bool(
                (self.run_rapps_on_start and self.orchestration_warmup_slots <= 0 and slot_id == 1)
                or slot_id == first_allowed_slot
            )
        else:
            should_run = (slot_id % self.rapp_interval_slots == 0)
        if force:
            should_run = True
        if not should_run:
            return False

        self.policy_store.start_new_epoch(slot_id=slot_id)
        current_epoch = int(getattr(self.policy_store, "current_epoch", 0))
        previous_reselection_reason = str(self._current_non_rt_reselection_reason or "")
        self._current_non_rt_reselection_reason = str(force_reason or "") if force else ""

        if self.active_intent_manager is not None:
            active_intent, intent_changed = self.active_intent_manager.get_epoch_state(current_epoch)
            if active_intent is not None:
                from models.policy import PolicyUpdate

                intent_update = PolicyUpdate(
                    active_intent_text=str(getattr(active_intent, "intent_text", "") or ""),
                    active_intent_id=str(getattr(active_intent, "intent_id", "") or ""),
                    intent_changed_this_epoch=bool(intent_changed),
                    translation_refreshed_this_epoch=False,
                    active_intent=active_intent,
                    policy_type_name="active_intent_state",
                    policy_description="Piecewise-constant active operator intent for the current Non-RT epoch",
                    source_name="active_intent_manager",
                )
                self.policy_store.apply_update(intent_update, slot_id=int(slot_id))
                self._emit_rapp_summary(slot_id, intent_update)

        for a1_interface in self.a1_interfaces:
            current_policy_snapshot = self.policy_store.get_snapshot()
            from ric.xapp_contract import is_rt, project, XAppContract
            rapp_observation = project(observation,XAppContract((),frozenset()),now_slot=slot_id)[0] if is_rt(observation) else observation
            from copy import deepcopy
            rapp_observation = deepcopy(rapp_observation)
            policy_update = a1_interface.send_observation(
                rapp_observation,
                policy_snapshot=current_policy_snapshot,
            )
            self.policy_store.register_ei_delivery(
                consumer_name=str(getattr(getattr(a1_interface, "rapp", None), "name", "unknown_rapp")),
                slot_id=int(slot_id),
                payload_summary=self._summarize_observation(observation),
                ei_type_name=str(
                    getattr(policy_update, "enrichment_info_type_name", "observation_context")
                ),
            )
            self.policy_store.apply_update(policy_update, slot_id=int(slot_id))
            self._emit_rapp_summary(slot_id, policy_update)
            debug_print(
                f"rApp triggered at slot {slot_id}. "
                f"source={policy_update.source_name}, "
                f"preferred_codebook_id={policy_update.preferred_codebook_id}, "
                f"preferred_beam_profile_id={policy_update.preferred_beam_profile_id}, "
                f"preferred_tx_power_dbm={policy_update.preferred_tx_power_dbm}, "
                f"preferred_scheduler_mode={policy_update.preferred_scheduler_mode}, "
                f"preferred_scheduler_policy_hint={policy_update.preferred_scheduler_policy_hint}, "
                f"preferred_latency_bias={policy_update.preferred_latency_bias}, "
                f"preferred_slice_weight_adjustment={policy_update.preferred_slice_weight_adjustment}, "
                f"preferred_beam_candidate_preference={policy_update.preferred_beam_candidate_preference}, "
                f"preferred_power_cap_dbm={policy_update.preferred_power_cap_dbm}, "
                f"preferred_cell_sleep_mode={policy_update.preferred_cell_sleep_mode}, "
                f"preferred_traffic_steering_preference={policy_update.preferred_traffic_steering_preference}, "
                f"preferred_mobility_bias_db={policy_update.preferred_mobility_bias_db}, "
                f"enabled_xapps={policy_update.enabled_xapps}, "
                f"disabled_xapps={policy_update.disabled_xapps}."
            )
        self._record_orchestration_epoch(
            slot_id=int(slot_id),
            observation=observation,
        )
        self._current_non_rt_reselection_reason = previous_reselection_reason
        self._initial_rapp_run_completed = True
        return True

    def _is_xapp_enabled(self, xapp_name: str) -> bool:
        """
        Checks whether one xApp is enabled according to the current policy epoch.
        INPUT
        xapp_name: The xApp name.
        OUTPUT
        True if the xApp is enabled, otherwise False.
        EXAMPLE OUTPUT
        True
        """
        if xapp_name in self.instance_selection.apps():return self.instance_selection.enabled(xapp_name)
        policy_snapshot = self.policy_store.get_snapshot()

        if xapp_name in policy_snapshot.disabled_xapps:
            return False

        if len(policy_snapshot.enabled_xapps) == 0:
            return True

        return xapp_name in policy_snapshot.enabled_xapps

    def _collect_xapp_outputs(
        self,
        slot_id: int,
        observation: Observation,
    ) -> tuple[
        Dict[str, Dict[str, Tuple[int, object, str]]],
        Dict[str, tuple[str, str | None]],
        list[CommandAction],
        list[PolicyConstraint],
        list[ConfigAction],
        list[DroppedControlItem],
    ]:
        """
        Runs all enabled Near-RT xApps when the interval condition is satisfied
        and collects winning field values based on field ownership and priority.
        INPUT
        slot_id: The current simulation slot.
        observation: The current aggregated observation.
        OUTPUT
        A dictionary that maps each action field to one tuple:
        (priority, chosen_value, xapp_name).
        EXAMPLE OUTPUT
        {
            "scheduler_mode": (10, "latency_aware", "latency_scheduler_xapp"),
            "codebook_id": (20, 2, "beam_xapp")
        }
        """
        self.rule_decisions=[]
        field_winners_by_scope: Dict[str, Dict[str, Tuple[int, object, str]]] = {}
        scope_specs_by_key: Dict[str, tuple[str, str | None]] = {}
        collected_commands: list[CommandAction] = []
        collected_constraints: list[PolicyConstraint] = []
        received_config_actions: list[ConfigAction] = []
        dropped_items: list[DroppedControlItem] = []

        if len(self.a1_interfaces) > 0 and not self._initial_rapp_run_completed:
            return (
                field_winners_by_scope,
                scope_specs_by_key,
                collected_commands,
                collected_constraints,
                received_config_actions,
                dropped_items,
            )

        if slot_id % self.xapp_interval_slots != 0:
            return (
                field_winners_by_scope,
                scope_specs_by_key,
                collected_commands,
                collected_constraints,
                received_config_actions,
                dropped_items,
            )

        self.instance_selection.sync(slot_id,observation)
        policy_snapshot = self.policy_store.get_snapshot()

        interfaces=self.e2_interfaces
        if self.instance_selection.mode=='conflict_experiment':
            interfaces=sorted(interfaces,key=lambda i:i.xapp.name,reverse=True)
        for e2_interface in interfaces:
            xapp = e2_interface.xapp

            if not self._is_xapp_enabled(xapp.name):
                debug_print(
                    f"xApp {xapp.name} is disabled in the current policy epoch at slot {slot_id}."
                )
                dropped_items.append(
                    DroppedControlItem(
                        kind="xapp",
                        source=str(xapp.name),
                        reason="disabled_in_policy_epoch",
                    )
                )
                continue

            self._record_e2_indication(
                subscriber_name=str(xapp.name),
                observation=observation,
                slot_id=int(slot_id),
            )
            from ric.xapp_contract import is_rt, project, validate_action, contract_for
            rt_contract = is_rt(observation)
            projected = observation
            contract = getattr(xapp, 'contract', contract_for(xapp.controlled_fields,xapp.name))
            if getattr(xapp,'library_instance',False):contract=xapp.runtime_contract(policy_snapshot)
            if rt_contract:
                projected, failures = project(observation, contract, self.xapp_interval_slots, slot_id)
                if not contract.available:
                    failures.append('no_supported_rt_actions')
                if failures:
                    dropped_items.append(DroppedControlItem(kind='xapp', source=str(xapp.name),
                        reason='kpi_contract_rejected:' + ';'.join(failures)))
                    continue
            from copy import deepcopy
            if getattr(xapp,'library_instance',False):
                # Independent control-plane receipt, never a simulation state reference.
                projected.control_ownership={}
                for (cell,field),owner in getattr(getattr(self,'rule_leases',None),'owned',{}).items():
                    projected.control_ownership.setdefault(cell,{})[field]=str(owner)
            xapp_decision = e2_interface.send_context(projected, deepcopy(policy_snapshot))
            if getattr(xapp,'rule_based_v1',False):
                self.rule_decisions=getattr(self,'rule_decisions',[])+deepcopy(xapp.last_decisions)
            if rt_contract:
                failures = [problem for item in xapp_decision.config_actions for problem in validate_action(item, contract, observation)]
                from ric.xapp_contract import validate_commands
                failures.extend(validate_commands(xapp_decision.commands,contract,observation))
                for command in xapp_decision.commands:
                    command._paper_allowed_cells=contract.cell_ids
                    command._paper_allowed_target_cells=contract.command_target_cell_ids or contract.cell_ids
                if failures:
                    dropped_items.append(DroppedControlItem(kind='xapp_action',source=str(xapp.name),
                        reason='action_contract_rejected:' + ';'.join(failures)))
                    continue
            for action_index,item_action in enumerate(xapp_decision.config_actions):
                xapp_action = item_action

                self._stamp_metadata(
                    xapp_action,
                    source=str(xapp.name),
                    slot_id=slot_id,
                    priority=int(xapp.priority),
                    reason="xapp_config_action",
                    decision_id_prefix="xapp_config",
                )
                received_config_actions.append(xapp_action)
                if not hasattr(self,'_field_validities'):self._field_validities={}
                validity=(xapp_action.effective_from_slot,xapp_action.effective_until_slot)
                scope_key, apply_scope, target_cell_id = self._normalize_scope_key(xapp_action)
                scope_specs_by_key[scope_key] = (apply_scope, target_cell_id)
                scope_field_winners = field_winners_by_scope.setdefault(scope_key, {})

                raw_commands = list(getattr(xapp_decision, "commands", [])) if action_index==0 else []
                for command in raw_commands:
                    self._stamp_metadata(
                        command,
                        source=str(xapp.name),
                        slot_id=slot_id,
                        priority=int(xapp.priority),
                        reason="xapp_command",
                        decision_id_prefix="xapp_cmd",
                    )
                    self._record_e2_control_request_for_command(
                        command=command,
                        source_name=str(xapp.name),
                        slot_id=int(slot_id),
                    )
                collected_commands.extend(raw_commands)

                raw_constraints = list(getattr(xapp_decision, "policy_constraints", [])) if action_index==0 else []
                for constraint in raw_constraints:
                    self._stamp_metadata(
                        constraint,
                        source=str(xapp.name),
                        slot_id=slot_id,
                        priority=int(xapp.priority),
                        reason="xapp_policy_constraint",
                        decision_id_prefix="xapp_constraint",
                    )
                collected_constraints.extend(raw_constraints)

                debug_print(
                    f"xApp triggered at slot {slot_id}. "
                    f"name={xapp.name}, "
                    f"priority={xapp.priority}, "
                    f"codebook_id={xapp_action.codebook_id}, "
                    f"beam_profile_id={xapp_action.beam_profile_id}, "
                    f"tx_power_dbm={xapp_action.tx_power_dbm}, "
                    f"scheduler_mode={xapp_action.scheduler_mode}, "
                    f"scheduler_policy_hint={xapp_action.scheduler_policy_hint}, "
                    f"latency_bias={xapp_action.latency_bias}, "
                    f"slice_weight_adjustment={xapp_action.slice_weight_adjustment}, "
                    f"beam_candidate_preference={xapp_action.beam_candidate_preference}, "
                    f"power_cap_dbm={xapp_action.power_cap_dbm}, "
                    f"cell_sleep_mode={xapp_action.cell_sleep_mode}, "
                    f"traffic_steering_preference={xapp_action.traffic_steering_preference}, "
                    f"mobility_bias_db={xapp_action.mobility_bias_db}, "
                    f"handover_cost_db={xapp_action.handover_cost_db}, "
                    f"handover_interruption_slots={xapp_action.handover_interruption_slots}, "
                    f"steering_hysteresis_db={xapp_action.steering_hysteresis_db}, "
                    f"measurement_event_trigger={xapp_action.measurement_event_trigger}, "
                    f"measurement_event_offset_db={xapp_action.measurement_event_offset_db}, "
                    f"commands={len(getattr(xapp_decision, 'commands', []))}, "
                    f"constraints={len(getattr(xapp_decision, 'policy_constraints', []))}."
                )

                for field_name in self.ACTION_FIELDS:
                    field_value = getattr(xapp_action, field_name)

                    if not self._action_field_has_value(field_name, field_value):
                        continue

                    if field_name not in xapp.controlled_fields:
                        debug_print(
                            f"xApp {xapp.name} tried to write undeclared field "
                            f"{field_name} at slot {slot_id}. This write is ignored."
                        )
                        dropped_items.append(
                            DroppedControlItem(
                                kind="field_write",
                                source=str(xapp.name),
                                reason="field_not_owned",
                                field_name=str(field_name),
                                value=self._copy_action_field_value(field_value),
                            )
                        )
                        continue

                    if field_name not in scope_field_winners:
                        scope_field_winners[field_name] = (
                            xapp.priority,
                            self._copy_action_field_value(field_value),
                            xapp.name,
                        )
                        self._field_validities[(scope_key,field_name)]=validity
                        continue

                    current_priority, current_value, current_owner = scope_field_winners[field_name]
                    if xapp.priority > current_priority:
                        dropped_items.append(DroppedControlItem(kind='field_write',source=str(current_owner),
                            reason='superseded_by_higher_priority',field_name=str(field_name),
                            value=self._copy_action_field_value(current_value)))
                        debug_print(
                            f"Field conflict at slot {slot_id}: "
                            f"{scope_key}.{field_name} reassigned from {current_owner} "
                            f"to {xapp.name} due to higher priority."
                        )
                        scope_field_winners[field_name] = (
                            xapp.priority,
                            self._copy_action_field_value(field_value),
                            xapp.name,
                        )
                        self._field_validities[(scope_key,field_name)]=validity
                    else:
                        dropped_items.append(
                            DroppedControlItem(
                                kind="field_write",
                                source=str(xapp.name),
                                reason="lower_priority_than_current_winner",
                                field_name=str(field_name),
                                value=self._copy_action_field_value(field_value),
                            )
                        )

        if any(getattr(i.xapp,'rule_based_v1',False) for i in self.e2_interfaces):
            from ric.near_rt.rule_based_v1.leases import RuleLeases
            if not hasattr(self,'rule_leases'):self.rule_leases=RuleLeases()
            self.rule_leases.reconcile(self,observation,field_winners_by_scope,scope_specs_by_key,received_config_actions,dropped_items)
        return (
            field_winners_by_scope,
            scope_specs_by_key,
            collected_commands,
            collected_constraints,
            received_config_actions,
            dropped_items,
        )

    def _near_rt_reselection_reason(
        self,
        *,
        slot_id: int,
        policy_constraints: list[PolicyConstraint],
    ) -> str:
        if not self.non_rt_reselection_enabled:
            return ""
        fallback_seen = False
        health_reasons: list[str] = []
        for constraint in list(policy_constraints or []):
            if str(getattr(constraint, "constraint_type", "") or "") != "goal_parameter_tuning_rl_joint_heads":
                continue
            params = dict(getattr(constraint, "params", {}) or {})
            if str(params.get("selection_mode", "") or "") == "safe_baseline_fallback":
                fallback_seen = True
            health_reasons.extend(
                str(item)
                for item in list(params.get("health_gate_block_reasons", []) or [])
                if str(item).strip() != ""
            )
        if fallback_seen or len(health_reasons) > 0:
            self._near_rt_fallback_streak += 1
        else:
            self._near_rt_fallback_streak = 0
            return ""
        if self._near_rt_fallback_streak < self.non_rt_reselection_fallback_threshold:
            return ""
        if int(slot_id) - int(self._last_forced_non_rt_reselection_slot) < self.non_rt_reselection_min_gap_slots:
            return ""
        self._last_forced_non_rt_reselection_slot = int(slot_id)
        self._near_rt_fallback_streak = 0
        reason_parts = ["near_rt_recovery"]
        if fallback_seen:
            reason_parts.append("safe_baseline_fallback")
        if len(health_reasons) > 0:
            reason_parts.append("health_gate=" + "|".join(sorted(set(health_reasons))[:3]))
        return ";".join(reason_parts)

    def _apply_field_winners(
        self,
        field_winners_by_scope: Dict[str, Dict[str, Tuple[int, object, str]]],
        scope_specs_by_key: Dict[str, tuple[str, str | None]],
    ) -> list[ConfigAction]:
        """
        Applies the winning field values into the current persistent action.
        INPUT
        field_winners: The winning field dictionary generated by xApps.
        OUTPUT
        Returns the merged persistent config-action list for all scopes.
        EXAMPLE OUTPUT
        self.current_action.scheduler_mode becomes "latency_aware"
        """
        for scope_key, scope_field_winners in field_winners_by_scope.items():
            apply_scope, target_cell_id = scope_specs_by_key.get(scope_key, ("cell", None))
            scope_action = self._build_scope_action(
                scope_key=scope_key,
                apply_scope=apply_scope,
                target_cell_id=target_cell_id,
            )
            if 'prb_power_w' in scope_field_winners:scope_action.clear_prb_power=None
            if scope_field_winners.get('clear_prb_power',(0,False,''))[1]:scope_action.prb_power_w=None
            for field_name, (_, field_value, _) in scope_field_winners.items():
                setattr(scope_action, field_name, self._copy_action_field_value(field_value))
            self.current_action = self.current_action_by_scope.get("cell:__default__", self.current_action)
            if scope_key == "cell:__default__":
                self.current_action = scope_action
                self._normalize_current_action_surface()
            else:
                current_default = self.current_action
                self.current_action = scope_action
                self._normalize_current_action_surface()
                self.current_action = current_default
        if "cell:__default__" in self.current_action_by_scope:
            self.current_action = self.current_action_by_scope["cell:__default__"]
        ordered_scope_keys = sorted(
            self.current_action_by_scope.keys(),
            key=lambda item: (0 if item == "cell:__default__" else 1, item),
        )
        grouped_actions=[]
        for scope_key in ordered_scope_keys:
            action=self.current_action_by_scope[scope_key]
            groups={}
            for field in self.ACTION_FIELDS:
                value=getattr(action,field)
                if not self._action_field_has_value(field,value):continue
                window=getattr(self,'_field_validities',{}).get((scope_key,field),(None,None))
                if window not in groups:
                    groups[window]=ConfigAction(apply_scope=action.apply_scope,target_cell_id=action.target_cell_id,
                        source_name=action.source_name,effective_from_slot=window[0],effective_until_slot=window[1])
                setattr(groups[window],field,self._copy_action_field_value(value))
            grouped_actions.extend(groups.values() if groups else [action])
        return grouped_actions

    def update(self, slot_id: int, observation: Observation) -> ControlDecision:
        """
        Updates Non-RT policies by epochs, then updates Near-RT executable
        actions, and returns the final executable action for the current slot.
        INPUT
        slot_id: The current simulation slot.
        observation: The current aggregated observation.
        OUTPUT
        One final executable Action object for the current slot.
        EXAMPLE OUTPUT
        Action(codebook_id=2, tx_power_dbm=31.0, beam_profile_id=1)
        """
        from ric.xapp_contract import is_rt, capability_record
        if is_rt(observation):
            catalog = [capability_record(i.xapp.name, i.xapp.controlled_fields,getattr(i.xapp,'contract',None)) for i in self.e2_interfaces]
            instances=self.instance_selection.apps()
            for item in catalog:
                if item['xapp_name'] in instances:item.update(instances[item['xapp_name']].descriptor())
            available = [item['xapp_name'] for item in catalog if item['available']]
            for interface in self.a1_interfaces:
                interface.rapp.rt_xapp_capabilities = catalog
                from ric.near_rt.xapp_library import category_catalog,implementation_catalog
                interface.rapp.rt_xapp_categories=category_catalog()
                interface.rapp.rt_xapp_implementations=implementation_catalog()
                interface.rapp.rt_xapp_instances=[app.descriptor() for app in instances.values()]
                interface.rapp.rt_backend = 'sionna_rt_sys'
                if hasattr(interface.rapp, 'available_xapp_names'):
                    interface.rapp.available_xapp_names = available
        self._run_rapps_if_needed(slot_id, observation)
        (
            field_winners_by_scope,
            scope_specs_by_key,
            commands,
            xapp_constraints,
            received_config_actions,
            dropped_items,
        ) = self._collect_xapp_outputs(slot_id, observation)
        reselection_reason = self._near_rt_reselection_reason(
            slot_id=int(slot_id),
            policy_constraints=list(xapp_constraints),
        )
        if reselection_reason:
            self._run_rapps_if_needed(
                slot_id,
                observation,
                force=True,
                force_reason=reselection_reason,
            )
            (
                field_winners_by_scope,
                scope_specs_by_key,
                commands,
                xapp_constraints,
                received_config_actions,
                dropped_items,
            ) = self._collect_xapp_outputs(slot_id, observation)
        merged_config_actions = self._apply_field_winners(
            field_winners_by_scope,
            scope_specs_by_key,
        )

        policy_snapshot = self.policy_store.get_snapshot()
        policy_constraints = list(getattr(policy_snapshot, "policy_constraints", []))
        policy_constraints.extend(xapp_constraints)

        for constraint in policy_constraints:
            self._stamp_metadata(
                constraint,
                source=str(getattr(constraint, "source_name", "unknown")),
                slot_id=slot_id,
                reason=str(getattr(getattr(constraint, "metadata", None), "reason", ""))
                or "active_policy_constraint",
                decision_id_prefix="active_constraint",
            )

        trace = ControlArbitrationTrace(
            received_config_actions=list(received_config_actions),
            merged_config_actions=list(merged_config_actions),
            accepted_commands=list(commands),
            active_constraints=list(policy_constraints),
            dropped_items_with_reason=list(dropped_items),
        )
        for scope_key, scope_field_winners in field_winners_by_scope.items():
            for field_name, (priority, field_value, owner) in scope_field_winners.items():
                scoped_field_name = (
                    f"{scope_key}.{field_name}"
                    if scope_key != "cell:__default__"
                    else str(field_name)
                )
                trace.selected_config_action_by_field[str(scoped_field_name)] = FieldArbitrationResult(
                    field_name=str(scoped_field_name),
                    value=field_value,
                    source=str(owner),
                    priority=int(priority),
                )

        debug_print(
            f"ControllerManager output at slot {slot_id}: "
            f"default_scope={self._summarize_config_action_for_debug(self.current_action)}, "
            f"merged_config_actions={len(merged_config_actions)} "
            f"{self._summarize_config_actions_for_debug(merged_config_actions)}, "
            f"commands={len(commands)}, "
            f"policy_constraints={len(policy_constraints)}."
        )

        stamped_config_actions: list[ConfigAction] = []
        for merged_scope_action in merged_config_actions:
            stamped_scope_action = ConfigAction(
                codebook_id=merged_scope_action.codebook_id,
                beam_profile_id=merged_scope_action.beam_profile_id,
                tx_power_dbm=merged_scope_action.tx_power_dbm,
                scheduler_mode=merged_scope_action.scheduler_mode,
                scheduler_policy_hint=merged_scope_action.scheduler_policy_hint,
                latency_bias=merged_scope_action.latency_bias,
                slice_weight_adjustment=dict(merged_scope_action.slice_weight_adjustment),
                beam_candidate_preference=merged_scope_action.beam_candidate_preference,
                power_cap_dbm=merged_scope_action.power_cap_dbm,
                cell_sleep_mode=merged_scope_action.cell_sleep_mode,
                traffic_steering_preference=merged_scope_action.traffic_steering_preference,
                mobility_bias_db=merged_scope_action.mobility_bias_db,
                handover_cost_db=merged_scope_action.handover_cost_db,
                handover_interruption_slots=merged_scope_action.handover_interruption_slots,
                steering_hysteresis_db=merged_scope_action.steering_hysteresis_db,
                measurement_event_trigger=merged_scope_action.measurement_event_trigger,
                measurement_event_offset_db=merged_scope_action.measurement_event_offset_db,
                apply_scope=getattr(merged_scope_action, "apply_scope", "cell"),
                target_cell_id=getattr(merged_scope_action, "target_cell_id", None),
                effective_from_slot=getattr(merged_scope_action, "effective_from_slot", None),
                effective_until_slot=getattr(merged_scope_action, "effective_until_slot", None),
                source_name="controller_manager",
            )
            from copy import deepcopy
            from phy.rt_sys.paper_controls import FIELDS as PAPER_FIELDS
            for field in PAPER_FIELDS:setattr(stamped_scope_action,field,deepcopy(getattr(merged_scope_action,field)))
            self._stamp_metadata(
                stamped_scope_action,
                source="controller_manager",
                slot_id=slot_id,
                reason="merged_config_action",
                decision_id_prefix="merged_config",
            )
            self._record_e2_control_request_for_config(
                action=stamped_scope_action,
                source_name="controller_manager",
                slot_id=int(slot_id),
            )
            if hasattr(self,'rule_leases'):
                scope='cell:'+str(stamped_scope_action.target_cell_id)
                stamped_scope_action._rule_owners=tuple(sorted({owner for (cid,field),owner in self.rule_leases.owned.items()
                    if cid==stamped_scope_action.target_cell_id and self._action_field_has_value(field,getattr(stamped_scope_action,field,None))}))
                apps=self.instance_selection.apps()
                stamped_scope_action._library_provenances=[{**apps[name].identity(stamped_scope_action.target_cell_id),
                    'fields':[field for (cid,field),owner in self.rule_leases.owned.items()
                        if owner==name and cid==stamped_scope_action.target_cell_id and self._action_field_has_value(field,getattr(stamped_scope_action,field,None))]}
                    for name in stamped_scope_action._rule_owners if name in apps]
            stamped_config_actions.append(stamped_scope_action)

        decision_metadata = ControlMetadata(
            source="controller_manager",
            issued_at_slot=int(slot_id),
            reason="merged_control_decision",
            decision_id=self._next_decision_id("decision"),
        )

        return ControlDecision(
            config_action=stamped_config_actions[0] if len(stamped_config_actions) > 0 else ConfigAction(),
            config_actions=list(stamped_config_actions),
            commands=list(commands),
            policy_constraints=list(policy_constraints),
            metadata=decision_metadata,
            trace=trace,
            oran_lifecycle_resources=self._build_oran_lifecycle_resources(),
        )
