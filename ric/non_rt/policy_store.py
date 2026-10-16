"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: policy_store.py
Description: This file defines the Non-RT policy store used by the
simulator. The policy store keeps the aggregated policy snapshot generated
within the current rApp policy epoch and exposes it to Near-RT xApps and
the controller manager.
"""

from __future__ import annotations

from core.debug import debug_print
from models.intent_orchestration import (
    active_intent_record_from_goal,
    orchestration_plan_view_from_plan,
    rl_guidance_contract_from_reward_spec,
    safety_constraint_set_from_plan,
    validated_intent_template_from_goal,
)
from models.oran_lifecycle import (
    A1EIJob,
    A1EIType,
    A1PolicyInstance,
    A1PolicyStatus,
    A1PolicyType,
    OranLifecycleResources,
)
from models.policy import PolicySnapshot, PolicyUpdate


class PolicyStore:
    """
    Stores the latest Non-RT policy snapshot.
    """

    def __init__(self) -> None:
        self.current_epoch = 0
        self.snapshot = PolicySnapshot(policy_epoch_id=self.current_epoch)
        self._resource_seq = 0
        self.derive_rl_guidance_when_missing = True
        self.binding_validator=None
        self.binding_receipts=[]

        self.policy_types_by_id: dict[str, A1PolicyType] = {}
        self.policy_type_id_by_name: dict[str, str] = {}
        self.policy_instances_by_id: dict[str, A1PolicyInstance] = {}
        self.policy_status_by_instance_id: dict[str, A1PolicyStatus] = {}
        self.latest_policy_instance_id_by_source: dict[str, str] = {}

        self.ei_types_by_id: dict[str, A1EIType] = {}
        self.ei_type_id_by_name: dict[str, str] = {}
        self.ei_jobs_by_id: dict[str, A1EIJob] = {}
        self.ei_job_id_by_consumer: dict[str, str] = {}

    def _next_resource_id(self, prefix: str) -> str:
        self._resource_seq += 1
        return f"{prefix}:{self._resource_seq}"

    def _get_or_create_policy_type(
        self,
        *,
        name: str,
        description: str = "",
        scope: str = "cluster",
    ) -> A1PolicyType:
        normalized_name = str(name).strip() or "generic_policy"
        existing_id = self.policy_type_id_by_name.get(normalized_name)
        if existing_id is not None and existing_id in self.policy_types_by_id:
            return self.policy_types_by_id[existing_id]

        policy_type = A1PolicyType(
            policy_type_id=self._next_resource_id("a1_policy_type"),
            name=normalized_name,
            description=str(description),
            scope=str(scope),
        )
        self.policy_types_by_id[policy_type.policy_type_id] = policy_type
        self.policy_type_id_by_name[normalized_name] = policy_type.policy_type_id
        return policy_type

    def _get_or_create_ei_type(
        self,
        *,
        name: str,
        description: str = "",
    ) -> A1EIType:
        normalized_name = str(name).strip() or "observation_context"
        existing_id = self.ei_type_id_by_name.get(normalized_name)
        if existing_id is not None and existing_id in self.ei_types_by_id:
            return self.ei_types_by_id[existing_id]

        ei_type = A1EIType(
            ei_type_id=self._next_resource_id("a1_ei_type"),
            name=normalized_name,
            description=str(description),
        )
        self.ei_types_by_id[ei_type.ei_type_id] = ei_type
        self.ei_type_id_by_name[normalized_name] = ei_type.ei_type_id
        return ei_type

    def _summarize_policy_update(self, policy_update: PolicyUpdate) -> dict[str, object]:
        non_default_fields = [
            field_name
            for field_name in [
                "preferred_codebook_id",
                "preferred_tx_power_dbm",
                "preferred_beam_profile_id",
                "preferred_scheduler_mode",
                "preferred_scheduler_policy_hint",
                "preferred_latency_bias",
                "preferred_beam_candidate_preference",
                "preferred_power_cap_dbm",
                "preferred_cell_sleep_mode",
                "preferred_traffic_steering_preference",
                "preferred_mobility_bias_db",
            ]
            if getattr(policy_update, field_name, None) is not None
        ]
        if getattr(policy_update, "preferred_slice_weight_adjustment", {}):
            non_default_fields.append("preferred_slice_weight_adjustment")

        return {
            "field_names": sorted(str(field_name) for field_name in non_default_fields),
            "enabled_xapp_count": int(len(getattr(policy_update, "enabled_xapps", []) or [])),
            "disabled_xapp_count": int(len(getattr(policy_update, "disabled_xapps", []) or [])),
            "constraint_count": int(len(getattr(policy_update, "policy_constraints", []) or [])),
            "has_active_intent": bool(getattr(policy_update, "active_intent", None) is not None),
            "has_validated_intent_template": bool(
                getattr(policy_update, "validated_intent_template", None) is not None
            ),
            "has_orchestration_plan_view": bool(
                getattr(policy_update, "orchestration_plan_view", None) is not None
            ),
            "has_rl_guidance": bool(getattr(policy_update, "rl_guidance", None) is not None),
            "has_safety_constraint_set": bool(
                getattr(policy_update, "safety_constraint_set", None) is not None
            ),
            "has_memory_state": bool(getattr(policy_update, "memory_state", None) is not None),
            "active_intent_id": str(getattr(policy_update, "active_intent_id", "") or ""),
            "intent_changed_this_epoch": getattr(policy_update, "intent_changed_this_epoch", None),
            "translation_refreshed_this_epoch": getattr(
                policy_update,
                "translation_refreshed_this_epoch",
                None,
            ),
        }

    def _supersede_previous_policy_instance(
        self,
        *,
        source_name: str,
        slot_id: int | None,
        reason: str,
    ) -> None:
        previous_instance_id = self.latest_policy_instance_id_by_source.get(str(source_name))
        if previous_instance_id is None:
            return
        previous_instance = self.policy_instances_by_id.get(previous_instance_id)
        previous_status = self.policy_status_by_instance_id.get(previous_instance_id)
        if previous_instance is not None:
            previous_instance.status = "superseded"
            previous_instance.updated_at_slot = slot_id
        if previous_status is not None:
            previous_status.status = "superseded"
            previous_status.reason = str(reason)
            previous_status.updated_at_slot = slot_id

    def start_new_epoch(self, slot_id: int | None = None) -> None:
        self.current_epoch += 1
        persistent_active_intent_text = self.snapshot.active_intent_text
        persistent_active_intent_id = self.snapshot.active_intent_id
        persistent_structured_goal = self.snapshot.structured_goal
        persistent_xapp_selection_plan = self.snapshot.xapp_selection_plan
        persistent_reward_spec = self.snapshot.reward_spec
        persistent_active_intent = self.snapshot.active_intent
        persistent_validated_intent_template = self.snapshot.validated_intent_template
        persistent_orchestration_plan_view = self.snapshot.orchestration_plan_view
        persistent_rl_guidance = self.snapshot.rl_guidance
        persistent_safety_constraint_set = self.snapshot.safety_constraint_set
        persistent_memory_state = self.snapshot.memory_state
        persistent_enabled_xapps = list(self.snapshot.enabled_xapps)
        persistent_disabled_xapps = list(self.snapshot.disabled_xapps)
        from copy import deepcopy
        persistent_bindings=deepcopy(self.snapshot.xapp_bindings)
        persistent_sets=deepcopy(self.snapshot.xapp_active_sets)
        persistent_selection_version=self.snapshot.xapp_selection_version
        persistent_rule_parameters=deepcopy(self.snapshot.rule_based_parameters) if self.binding_validator else {}
        self.snapshot = PolicySnapshot(policy_epoch_id=self.current_epoch)
        self.snapshot.active_intent_text = persistent_active_intent_text
        self.snapshot.active_intent_id = persistent_active_intent_id
        self.snapshot.intent_changed_this_epoch = False
        self.snapshot.translation_refreshed_this_epoch = False
        self.snapshot.structured_goal = persistent_structured_goal
        self.snapshot.xapp_selection_plan = persistent_xapp_selection_plan
        self.snapshot.reward_spec = persistent_reward_spec
        self.snapshot.active_intent = persistent_active_intent
        self.snapshot.validated_intent_template = persistent_validated_intent_template
        self.snapshot.orchestration_plan_view = persistent_orchestration_plan_view
        self.snapshot.rl_guidance = persistent_rl_guidance
        self.snapshot.safety_constraint_set = persistent_safety_constraint_set
        self.snapshot.memory_state = persistent_memory_state
        self.snapshot.enabled_xapps = persistent_enabled_xapps
        self.snapshot.disabled_xapps = persistent_disabled_xapps
        self.snapshot.xapp_bindings=persistent_bindings
        self.snapshot.xapp_active_sets=persistent_sets
        self.snapshot.xapp_selection_version=persistent_selection_version
        self.snapshot.rule_based_parameters=persistent_rule_parameters
        for instance_id, instance in self.policy_instances_by_id.items():
            if str(instance.status) != "active":
                continue
            instance.status = "superseded"
            instance.updated_at_slot = slot_id
            status = self.policy_status_by_instance_id.get(instance_id)
            if status is not None:
                status.status = "superseded"
                status.reason = "policy_epoch_reset"
                status.updated_at_slot = slot_id
        debug_print("PolicyStore started a new policy epoch.")

    def register_ei_delivery(
        self,
        *,
        consumer_name: str,
        slot_id: int,
        ei_type_name: str = "observation_context",
        payload_summary: dict[str, object] | None = None,
    ) -> A1EIJob:
        ei_type = self._get_or_create_ei_type(
            name=ei_type_name,
            description="A1-EI-like observation delivery",
        )
        job_id = self.ei_job_id_by_consumer.get(str(consumer_name))
        if job_id is None or job_id not in self.ei_jobs_by_id:
            job = A1EIJob(
                job_id=self._next_resource_id("a1_ei_job"),
                ei_type_id=ei_type.ei_type_id,
                consumer_name=str(consumer_name),
                created_at_slot=int(slot_id),
                updated_at_slot=int(slot_id),
                last_delivery_slot=int(slot_id),
                status="active",
                payload_summary=dict(payload_summary or {}),
            )
            self.ei_jobs_by_id[job.job_id] = job
            self.ei_job_id_by_consumer[str(consumer_name)] = job.job_id
            return job

        job = self.ei_jobs_by_id[job_id]
        job.status = "active"
        job.updated_at_slot = int(slot_id)
        job.last_delivery_slot = int(slot_id)
        job.payload_summary = dict(payload_summary or {})
        return job

    def apply_update(self, policy_update: PolicyUpdate, *, slot_id: int | None = None) -> None:
        from copy import deepcopy
        legacy_names=list(policy_update.enabled_xapps or [])+list(policy_update.disabled_xapps or [])
        if policy_update.xapp_active_sets is not None or policy_update.xapp_bindings is not None or (self.binding_validator and (policy_update.rule_based_parameters or legacy_names)):
            try:
                if self.binding_validator is None:raise ValueError('No instance library configured')
                candidate=self.snapshot.xapp_bindings if policy_update.xapp_bindings is None else policy_update.xapp_bindings
                active_sets=self.snapshot.xapp_active_sets if policy_update.xapp_active_sets is None else policy_update.xapp_active_sets
                if self.binding_validator.__self__.mode=='exclusive':candidate=candidate or {}
                elif active_sets is None:active_sets={}
                self.binding_validator(candidate,policy_update.rule_based_parameters,legacy_names,active_sets=active_sets)
            except (ValueError,TypeError) as error:
                self.binding_receipts.append(dict(slot_id=slot_id,kind='xapp_selection',status='rejected',reason=str(error),source=policy_update.source_name))
                return False
        if policy_update.xapp_bindings is not None:
            self.snapshot.xapp_bindings=deepcopy(policy_update.xapp_bindings)
            self.binding_receipts.append(dict(slot_id=slot_id,kind='xapp_selection',status='admitted',reason='valid_pending_near_rt',source=policy_update.source_name))
        if policy_update.xapp_active_sets is not None:
            self.snapshot.xapp_active_sets=deepcopy(policy_update.xapp_active_sets)
            self.binding_receipts.append(dict(slot_id=slot_id,kind='xapp_selection',status='admitted',reason='valid_active_sets_pending_near_rt',source=policy_update.source_name))
        self.snapshot.rule_based_parameters.update(deepcopy(policy_update.rule_based_parameters))
        source_name = str(getattr(policy_update, "source_name", "unknown_rapp"))
        policy_type = self._get_or_create_policy_type(
            name=str(getattr(policy_update, "policy_type_name", "") or source_name),
            description=str(getattr(policy_update, "policy_description", "")),
            scope="cluster",
        )
        self._supersede_previous_policy_instance(
            source_name=source_name,
            slot_id=slot_id,
            reason="replaced_by_new_policy_update",
        )

        policy_instance = A1PolicyInstance(
            instance_id=self._next_resource_id("a1_policy_instance"),
            policy_type_id=policy_type.policy_type_id,
            source_name=source_name,
            created_at_slot=slot_id,
            updated_at_slot=slot_id,
            status="active",
            target_xapps=list(getattr(policy_update, "enabled_xapps", []) or []),
            payload_summary=self._summarize_policy_update(policy_update),
        )
        policy_status = A1PolicyStatus(
            instance_id=policy_instance.instance_id,
            status="active",
            reason="policy_update_applied_to_epoch",
            updated_at_slot=slot_id,
        )
        self.policy_instances_by_id[policy_instance.instance_id] = policy_instance
        self.policy_status_by_instance_id[policy_instance.instance_id] = policy_status
        self.latest_policy_instance_id_by_source[source_name] = policy_instance.instance_id

        if policy_update.policy_epoch_id is not None:
            self.snapshot.policy_epoch_id = int(policy_update.policy_epoch_id)

        if policy_update.preferred_codebook_id is not None:
            self.snapshot.preferred_codebook_id = policy_update.preferred_codebook_id

        if policy_update.preferred_tx_power_dbm is not None:
            self.snapshot.preferred_tx_power_dbm = policy_update.preferred_tx_power_dbm

        if policy_update.preferred_beam_profile_id is not None:
            self.snapshot.preferred_beam_profile_id = policy_update.preferred_beam_profile_id

        if policy_update.preferred_scheduler_mode is not None:
            self.snapshot.preferred_scheduler_mode = policy_update.preferred_scheduler_mode

        if policy_update.preferred_scheduler_policy_hint is not None:
            self.snapshot.preferred_scheduler_policy_hint = policy_update.preferred_scheduler_policy_hint

        if policy_update.preferred_latency_bias is not None:
            self.snapshot.preferred_latency_bias = float(policy_update.preferred_latency_bias)

        if policy_update.preferred_slice_weight_adjustment:
            self.snapshot.preferred_slice_weight_adjustment.update(
                dict(policy_update.preferred_slice_weight_adjustment)
            )

        if policy_update.preferred_beam_candidate_preference is not None:
            self.snapshot.preferred_beam_candidate_preference = policy_update.preferred_beam_candidate_preference

        if policy_update.preferred_power_cap_dbm is not None:
            self.snapshot.preferred_power_cap_dbm = float(policy_update.preferred_power_cap_dbm)

        if policy_update.preferred_cell_sleep_mode is not None:
            self.snapshot.preferred_cell_sleep_mode = str(policy_update.preferred_cell_sleep_mode)

        if policy_update.preferred_traffic_steering_preference is not None:
            self.snapshot.preferred_traffic_steering_preference = (
                policy_update.preferred_traffic_steering_preference
            )

        if policy_update.preferred_mobility_bias_db is not None:
            self.snapshot.preferred_mobility_bias_db = float(policy_update.preferred_mobility_bias_db)

        if policy_update.preferred_handover_cost_db is not None:
            self.snapshot.preferred_handover_cost_db = float(policy_update.preferred_handover_cost_db)

        if policy_update.preferred_handover_interruption_slots is not None:
            self.snapshot.preferred_handover_interruption_slots = int(
                policy_update.preferred_handover_interruption_slots
            )

        if policy_update.preferred_steering_hysteresis_db is not None:
            self.snapshot.preferred_steering_hysteresis_db = float(
                policy_update.preferred_steering_hysteresis_db
            )

        if policy_update.preferred_measurement_event_trigger is not None:
            self.snapshot.preferred_measurement_event_trigger = str(
                policy_update.preferred_measurement_event_trigger
            )

        if policy_update.preferred_measurement_event_offset_db is not None:
            self.snapshot.preferred_measurement_event_offset_db = float(
                policy_update.preferred_measurement_event_offset_db
            )

        if policy_update.active_intent_text is not None:
            self.snapshot.active_intent_text = str(policy_update.active_intent_text)

        if policy_update.active_intent_id is not None:
            self.snapshot.active_intent_id = str(policy_update.active_intent_id)

        if policy_update.intent_changed_this_epoch is not None:
            self.snapshot.intent_changed_this_epoch = bool(policy_update.intent_changed_this_epoch)

        if policy_update.translation_refreshed_this_epoch is not None:
            self.snapshot.translation_refreshed_this_epoch = bool(
                policy_update.translation_refreshed_this_epoch
            )

        if policy_update.structured_goal is not None:
            self.snapshot.structured_goal = policy_update.structured_goal

        if policy_update.xapp_selection_plan is not None:
            self.snapshot.xapp_selection_plan = policy_update.xapp_selection_plan

        if policy_update.reward_spec is not None:
            self.snapshot.reward_spec = policy_update.reward_spec

        if policy_update.active_intent is not None:
            self.snapshot.active_intent = policy_update.active_intent

        if policy_update.validated_intent_template is not None:
            self.snapshot.validated_intent_template = policy_update.validated_intent_template

        if policy_update.orchestration_plan_view is not None:
            self.snapshot.orchestration_plan_view = policy_update.orchestration_plan_view

        if policy_update.rl_guidance is not None:
            self.snapshot.rl_guidance = policy_update.rl_guidance

        if policy_update.safety_constraint_set is not None:
            self.snapshot.safety_constraint_set = policy_update.safety_constraint_set

        if policy_update.memory_state is not None:
            self.snapshot.memory_state = policy_update.memory_state

        should_refresh_active_intent = self.snapshot.active_intent is None
        if not should_refresh_active_intent and policy_update.active_intent_id is not None:
            should_refresh_active_intent = (
                str(getattr(self.snapshot.active_intent, "intent_id", "") or "")
                != str(policy_update.active_intent_id or "")
            )
        if not should_refresh_active_intent and policy_update.active_intent_text is not None:
            should_refresh_active_intent = (
                str(getattr(self.snapshot.active_intent, "intent_text", "") or "")
                != str(policy_update.active_intent_text or "")
            )
        if should_refresh_active_intent:
            derived_active_intent = active_intent_record_from_goal(
                self.snapshot.structured_goal,
                current_epoch=self.current_epoch,
                intent_text=str(self.snapshot.active_intent_text or ""),
                change_reason="legacy_policy_projection",
                source_name=source_name,
            )
            if derived_active_intent is not None:
                self.snapshot.active_intent = derived_active_intent
        if self.snapshot.active_intent_id is None and self.snapshot.active_intent is not None:
            self.snapshot.active_intent_id = str(getattr(self.snapshot.active_intent, "intent_id", "") or "")
        if self.snapshot.active_intent_id is None and self.snapshot.structured_goal is not None:
            self.snapshot.active_intent_id = str(getattr(self.snapshot.structured_goal, "intent_id", "") or "")

        if self.snapshot.validated_intent_template is None and self.snapshot.structured_goal is not None:
            self.snapshot.validated_intent_template = validated_intent_template_from_goal(
                self.snapshot.structured_goal,
                validator_name="legacy_policy_projection",
                projector_name="legacy_policy_projection",
            )

        if self.snapshot.safety_constraint_set is None and self.snapshot.xapp_selection_plan is not None:
            self.snapshot.safety_constraint_set = safety_constraint_set_from_plan(
                self.snapshot.xapp_selection_plan,
                generated_by="legacy_policy_projection",
            )

        if self.snapshot.orchestration_plan_view is None and self.snapshot.xapp_selection_plan is not None:
            self.snapshot.orchestration_plan_view = orchestration_plan_view_from_plan(
                self.snapshot.xapp_selection_plan,
            )

        if (
            self.derive_rl_guidance_when_missing
            and self.snapshot.rl_guidance is None
            and self.snapshot.reward_spec is not None
        ):
            derived_from_intent_id = ""
            if self.snapshot.active_intent is not None:
                derived_from_intent_id = str(getattr(self.snapshot.active_intent, "intent_id", "") or "")
            elif self.snapshot.structured_goal is not None:
                derived_from_intent_id = str(getattr(self.snapshot.structured_goal, "intent_id", "") or "")
            self.snapshot.rl_guidance = rl_guidance_contract_from_reward_spec(
                self.snapshot.reward_spec,
                selected_safety_rule_set=self.snapshot.safety_constraint_set,
                source_epoch=self.snapshot.policy_epoch_id,
                derived_from_intent_id=derived_from_intent_id,
                fallback_used=False,
            )

        for xapp_name in policy_update.enabled_xapps:
            if xapp_name in self.snapshot.disabled_xapps:
                self.snapshot.disabled_xapps.remove(xapp_name)
            if xapp_name not in self.snapshot.enabled_xapps:
                self.snapshot.enabled_xapps.append(xapp_name)

        for xapp_name in policy_update.disabled_xapps:
            if xapp_name in self.snapshot.enabled_xapps:
                self.snapshot.enabled_xapps.remove(xapp_name)
            if xapp_name not in self.snapshot.disabled_xapps:
                self.snapshot.disabled_xapps.append(xapp_name)

        for constraint in getattr(policy_update, "policy_constraints", []):
            metadata = getattr(constraint, "metadata", None)
            if metadata is not None and getattr(metadata, "source", "unknown") == "unknown":
                metadata.source = source_name
            if hasattr(constraint, "source_name") and getattr(constraint, "source_name") == "unknown":
                constraint.source_name = source_name
            if constraint not in self.snapshot.policy_constraints:
                self.snapshot.policy_constraints.append(constraint)

        debug_print(
            f"PolicyStore updated by {policy_update.source_name}. "
            f"preferred_scheduler_hint={self.snapshot.preferred_scheduler_policy_hint}, "
            f"preferred_latency_bias={self.snapshot.preferred_latency_bias}, "
            f"preferred_slice_weight_adjustment={self.snapshot.preferred_slice_weight_adjustment}, "
            f"preferred_beam_candidate_preference={self.snapshot.preferred_beam_candidate_preference}, "
            f"preferred_power_cap_dbm={self.snapshot.preferred_power_cap_dbm}, "
            f"preferred_cell_sleep_mode={self.snapshot.preferred_cell_sleep_mode}, "
            f"preferred_traffic_steering_preference={self.snapshot.preferred_traffic_steering_preference}, "
            f"preferred_mobility_bias_db={self.snapshot.preferred_mobility_bias_db}, "
            f"preferred_handover_cost_db={self.snapshot.preferred_handover_cost_db}, "
            f"preferred_handover_interruption_slots={self.snapshot.preferred_handover_interruption_slots}, "
            f"preferred_steering_hysteresis_db={self.snapshot.preferred_steering_hysteresis_db}, "
            f"preferred_measurement_event_trigger={self.snapshot.preferred_measurement_event_trigger}, "
            f"preferred_measurement_event_offset_db={self.snapshot.preferred_measurement_event_offset_db}."
        )

    def get_snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(
            policy_epoch_id=int(self.snapshot.policy_epoch_id),
            xapp_bindings=__import__('copy').deepcopy(self.snapshot.xapp_bindings),
            xapp_active_sets=__import__('copy').deepcopy(self.snapshot.xapp_active_sets),
            xapp_selection_version=self.snapshot.xapp_selection_version,
            rule_based_parameters=__import__("copy").deepcopy(self.snapshot.rule_based_parameters),
            preferred_codebook_id=self.snapshot.preferred_codebook_id,
            preferred_tx_power_dbm=self.snapshot.preferred_tx_power_dbm,
            preferred_beam_profile_id=self.snapshot.preferred_beam_profile_id,
            preferred_scheduler_mode=self.snapshot.preferred_scheduler_mode,
            preferred_scheduler_policy_hint=self.snapshot.preferred_scheduler_policy_hint,
            preferred_latency_bias=self.snapshot.preferred_latency_bias,
            preferred_slice_weight_adjustment=dict(self.snapshot.preferred_slice_weight_adjustment),
            preferred_beam_candidate_preference=self.snapshot.preferred_beam_candidate_preference,
            preferred_power_cap_dbm=self.snapshot.preferred_power_cap_dbm,
            preferred_cell_sleep_mode=self.snapshot.preferred_cell_sleep_mode,
            preferred_traffic_steering_preference=self.snapshot.preferred_traffic_steering_preference,
            preferred_mobility_bias_db=self.snapshot.preferred_mobility_bias_db,
            preferred_handover_cost_db=self.snapshot.preferred_handover_cost_db,
            preferred_handover_interruption_slots=self.snapshot.preferred_handover_interruption_slots,
            preferred_steering_hysteresis_db=self.snapshot.preferred_steering_hysteresis_db,
            preferred_measurement_event_trigger=self.snapshot.preferred_measurement_event_trigger,
            preferred_measurement_event_offset_db=self.snapshot.preferred_measurement_event_offset_db,
            active_intent_text=self.snapshot.active_intent_text,
            active_intent_id=self.snapshot.active_intent_id,
            intent_changed_this_epoch=bool(self.snapshot.intent_changed_this_epoch),
            translation_refreshed_this_epoch=bool(self.snapshot.translation_refreshed_this_epoch),
            structured_goal=self.snapshot.structured_goal,
            xapp_selection_plan=self.snapshot.xapp_selection_plan,
            reward_spec=self.snapshot.reward_spec,
            active_intent=self.snapshot.active_intent,
            validated_intent_template=self.snapshot.validated_intent_template,
            orchestration_plan_view=self.snapshot.orchestration_plan_view,
            rl_guidance=self.snapshot.rl_guidance,
            safety_constraint_set=self.snapshot.safety_constraint_set,
            memory_state=self.snapshot.memory_state,
            enabled_xapps=list(self.snapshot.enabled_xapps),
            disabled_xapps=list(self.snapshot.disabled_xapps),
            policy_constraints=list(self.snapshot.policy_constraints),
        )

    def get_lifecycle_resources(self) -> OranLifecycleResources:
        return OranLifecycleResources(
            a1_policy_types=list(self.policy_types_by_id.values()),
            a1_policy_instances=list(self.policy_instances_by_id.values()),
            a1_policy_statuses=list(self.policy_status_by_instance_id.values()),
            a1_ei_types=list(self.ei_types_by_id.values()),
            a1_ei_jobs=list(self.ei_jobs_by_id.values()),
        )
