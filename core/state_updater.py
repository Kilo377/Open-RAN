"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: state_updater.py
Description: This file defines the state updater used by the simulator.
It applies PHY results to UE state, records snapshots, and records slot
metrics. This implementation is compatible with both list-style and
dict-style state.ues containers.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from core.debug import debug_print
from core.metrics import build_observation
from core.runtime_views import (
    get_cluster_grants,
    get_cluster_resource_grid_summary,
    get_cluster_scheduled_ue_ids,
    get_cluster_scheduler_demands,
    get_cluster_scheduler_stage_trace,
)
from state.harq_state import HarqLogicalChannelState, HarqProcessState
from state.logical_channel_group_state import LogicalChannelGroupState
from state.resource_grid_state import ResourceAllocationState, ResourceGridState


class StateUpdater:
    """
    L1/L2 Design:
    This class updates simulator state according to PHY results and stores
    per-slot snapshot / metric history.
    """

    _HARQ_RV_SEQUENCE = (0, 2, 3, 1)

    def __init__(self) -> None:
        """
        Initializes the StateUpdater.
        """
        pass

    def _iter_ues(self, state) -> List[Any]:
        """
        Returns UE objects as a list regardless of whether state.ues is a
        dict or a list.
        INPUT
        state: Simulator state object.
        OUTPUT
        List of UE objects.
        """
        ues = getattr(state, "ues", [])
        if isinstance(ues, dict):
            return list(ues.values())
        if isinstance(ues, list):
            return ues
        return []

    def _find_ue_by_id(self, state, ue_id: str) -> Optional[Any]:
        """
        Finds a UE object by ue_id regardless of whether state.ues is a dict
        or a list.
        INPUT
        state: Simulator state object.
        ue_id: UE identifier string.
        OUTPUT
        UE object or None.
        """
        ues = getattr(state, "ues", [])

        if isinstance(ues, dict):
            return ues.get(ue_id)

        if isinstance(ues, list):
            for ue in ues:
                if getattr(ue, "ue_id", None) == ue_id:
                    return ue

        return None

    def _safe_get(self, obj: Any, field_name: str, default: Any) -> Any:
        """
        Safely gets an attribute from an object.
        INPUT
        obj: Any object.
        field_name: Attribute name.
        default: Default value if missing.
        OUTPUT
        Attribute value or default.
        """
        return getattr(obj, field_name, default)

    def _safe_set(self, obj: Any, field_name: str, value: Any) -> None:
        """
        Safely sets an attribute on an object.
        INPUT
        obj: Any object.
        field_name: Attribute name.
        value: Value to set.
        OUTPUT
        None.
        """
        try:
            setattr(obj, field_name, value)
        except Exception:
            pass

    def _get_connection(self, state: Any, ue_id: str) -> Any | None:
        connections = getattr(state, "connections", {})
        if isinstance(connections, dict):
            return connections.get(str(ue_id))
        return None

    def _get_link(self, state: Any, ue_id: str) -> Any | None:
        links = getattr(state, "links", {})
        if isinstance(links, dict):
            return links.get(str(ue_id))
        return None

    def _get_beamforming(self, state: Any, ue_id: str) -> Any | None:
        beamforming_map = getattr(state, "beamforming", {})
        if isinstance(beamforming_map, dict):
            return beamforming_map.get(str(ue_id))
        return None

    def _mean_or_none(self, values: List[Any]) -> float | None:
        normalized_values = []
        for value in values:
            if value is None:
                continue
            try:
                normalized_values.append(float(value))
            except (TypeError, ValueError):
                continue

        if len(normalized_values) == 0:
            return None

        return float(sum(normalized_values) / float(len(normalized_values)))

    def _rl_tuning_constraint_summary(self, control_state: Any | None) -> dict[str, Any]:
        if control_state is None:
            return {}
        constraints = list(getattr(control_state, "active_policy_constraints", []) or [])
        for constraint in constraints:
            constraint_type = str(getattr(constraint, "constraint_type", "") or "")
            if constraint_type != "goal_parameter_tuning_rl_joint_heads":
                continue
            params = dict(getattr(constraint, "params", {}) or {})
            selection_mode = str(params.get("selection_mode", "") or "")
            return {
                "rl_guidance_mode": str(params.get("guidance_mode", "") or ""),
                "rl_guidance_id": str(params.get("guidance_id", "") or ""),
                "rl_guidance_source_epoch": params.get("guidance_source_epoch", None),
                "rl_selection_mode": selection_mode,
                "rl_safe_baseline_fallback": bool(selection_mode == "safe_baseline_fallback"),
                "rl_health_gate_block_reasons": list(
                    params.get("health_gate_block_reasons", []) or []
                ),
                "rl_selected_safety_rule_ids": [
                    str(item)
                    for item in list(params.get("selected_safety_rule_ids", []) or [])
                    if str(item).strip() != ""
                ],
            }
        return {}

    def _effective_control_profile_summary(self, state: Any) -> dict[str, Any]:
        managed_cells = getattr(state, "managed_cells", None)
        profile_by_cell: dict[str, dict[str, Any]] = {}

        if isinstance(managed_cells, dict) and len(managed_cells) > 0:
            scoped_states = list(managed_cells.values())
        else:
            scoped_states = [state]

        for scoped_state in scoped_states:
            cell_state = getattr(scoped_state, "cell", None)
            scheduler_state = getattr(scoped_state, "scheduler", None)
            cell_id = str(getattr(cell_state, "cell_id", "cell_0"))
            slice_weight_adjustment = dict(
                getattr(scheduler_state, "slice_weight_adjustment", {}) or {}
            )
            profile_by_cell[cell_id] = {
                "scheduler_policy_hint": getattr(
                    scheduler_state, "scheduler_policy_hint", None
                ),
                "latency_bias": getattr(scheduler_state, "latency_bias", None),
                "slice_weight_adjustment": {
                    str(slice_id): float(weight)
                    for slice_id, weight in sorted(slice_weight_adjustment.items())
                },
                "beam_candidate_preference": getattr(
                    cell_state, "beam_candidate_preference", None
                ),
                "power_cap_dbm": getattr(cell_state, "power_cap_dbm", None),
                "traffic_steering_preference": getattr(
                    cell_state, "traffic_steering_preference", None
                ),
                "mobility_bias_db": getattr(cell_state, "mobility_bias_db", None),
                "measurement_event_offset_db": getattr(
                    cell_state, "measurement_event_offset_db", None
                ),
            }

        return {"cells": profile_by_cell}

    def _get_logical_channels(self, ue: Any) -> List[Any]:
        logical_channels = list(self._safe_get(ue, "logical_channels", []) or [])
        return [
            logical_channel
            for logical_channel in logical_channels
            if str(getattr(logical_channel, "logical_channel_id", "")) != ""
        ]

    def _get_logical_channel(self, ue: Any, logical_channel_id: str) -> Any | None:
        for logical_channel in self._get_logical_channels(ue):
            if str(getattr(logical_channel, "logical_channel_id", "")) == str(logical_channel_id):
                return logical_channel
        return None

    def _get_logical_channel_groups(self, ue: Any) -> List[Any]:
        logical_channel_groups = list(self._safe_get(ue, "logical_channel_groups", []) or [])
        return [
            logical_channel_group
            for logical_channel_group in logical_channel_groups
            if getattr(logical_channel_group, "lcg_id", None) is not None
        ]

    def _recompute_logical_channel_groups(
        self,
        ue: Any,
        *,
        slot_id: int | None = None,
    ) -> List[LogicalChannelGroupState]:
        logical_channels = self._get_logical_channels(ue)
        current_slot_id = None if slot_id is None else int(slot_id)
        if len(logical_channels) == 0:
            logical_channel_groups = self._get_logical_channel_groups(ue)
            for logical_channel_group in logical_channel_groups:
                if current_slot_id is None:
                    continue
                last_bsr_report_slot = getattr(logical_channel_group, "last_bsr_report_slot", None)
                logical_channel_group.bsr_age_slots = (
                    max(0, current_slot_id - int(last_bsr_report_slot))
                    if last_bsr_report_slot is not None
                    else None
                )
                logical_channel_group.bsr_triggered = False
            self._safe_set(ue, "logical_channel_groups", list(logical_channel_groups))
            return list(logical_channel_groups)

        previous_groups_by_lcg_id = {
            int(getattr(logical_channel_group, "lcg_id", 0)): logical_channel_group
            for logical_channel_group in self._get_logical_channel_groups(ue)
        }
        groups_by_lcg_id: dict[int, LogicalChannelGroupState] = {}
        for logical_channel in logical_channels:
            logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
            if logical_channel_id == "":
                continue
            lcg_id = int(getattr(logical_channel, "lcg_id", 0))
            previous_group = previous_groups_by_lcg_id.get(lcg_id)
            logical_channel_group = groups_by_lcg_id.setdefault(
                lcg_id,
                LogicalChannelGroupState(
                    lcg_id=lcg_id,
                    bsr_reported_bytes=int(
                        getattr(previous_group, "bsr_reported_bytes", 0)
                    ) if previous_group is not None else 0,
                    bsr_pending_bytes=int(
                        getattr(previous_group, "bsr_pending_bytes", 0)
                    ) if previous_group is not None else 0,
                    bsr_age_slots=getattr(previous_group, "bsr_age_slots", None)
                    if previous_group is not None else None,
                    last_bsr_report_slot=getattr(previous_group, "last_bsr_report_slot", None)
                    if previous_group is not None else None,
                    last_bsr_trigger_slot=getattr(previous_group, "last_bsr_trigger_slot", None)
                    if previous_group is not None else None,
                    bsr_triggered=False,
                    bsr_trigger_reason=getattr(previous_group, "bsr_trigger_reason", None)
                    if previous_group is not None else None,
                    bsr_periodic_timer_slots=int(
                        getattr(previous_group, "bsr_periodic_timer_slots", 5)
                    ) if previous_group is not None else 5,
                ),
            )
            logical_channel_group.logical_channel_ids.append(logical_channel_id)
            drb_id = getattr(logical_channel, "drb_id", None)
            if drb_id is not None:
                logical_channel_group.drb_ids.append(int(drb_id))
            queue_bytes = int(getattr(logical_channel, "queue_bytes", 0))
            logical_channel_group.queue_bytes += int(queue_bytes)
            if getattr(logical_channel, "prioritized_bit_rate_bps", None) is not None:
                logical_channel_group.prioritized_queue_bytes += int(queue_bytes)
            logical_channel_group.pbr_credit_bytes += float(
                getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0
            )
            logical_channel_group.scheduled_bytes += int(
                getattr(logical_channel, "scheduled_bytes", 0)
            )
            logical_channel_group.prioritized_scheduled_bytes += int(
                getattr(logical_channel, "prioritized_scheduled_bytes", 0)
            )
            logical_channel_group.delivered_bytes += int(
                getattr(logical_channel, "delivered_bytes", 0)
            )

        logical_channel_groups = sorted(
            groups_by_lcg_id.values(),
            key=lambda logical_channel_group: int(logical_channel_group.lcg_id),
        )
        for logical_channel_group in logical_channel_groups:
            logical_channel_group.logical_channel_ids = sorted(
                set(logical_channel_group.logical_channel_ids)
            )
            logical_channel_group.drb_ids = sorted(set(logical_channel_group.drb_ids))
            previous_group = previous_groups_by_lcg_id.get(int(logical_channel_group.lcg_id))
            previous_queue_bytes = (
                int(getattr(previous_group, "queue_bytes", 0))
                if previous_group is not None
                else 0
            )
            previous_bsr_reported_bytes = (
                int(getattr(previous_group, "bsr_reported_bytes", 0))
                if previous_group is not None
                else 0
            )
            last_bsr_report_slot = getattr(logical_channel_group, "last_bsr_report_slot", None)
            periodic_timer_slots = max(
                0,
                int(getattr(logical_channel_group, "bsr_periodic_timer_slots", 5)),
            )
            periodic_timer_expired = (
                current_slot_id is not None
                and last_bsr_report_slot is not None
                and periodic_timer_slots > 0
                and (current_slot_id - int(last_bsr_report_slot)) >= periodic_timer_slots
                and int(logical_channel_group.queue_bytes) > 0
            )
            trigger_reason = None
            if current_slot_id is not None:
                if previous_group is None and int(logical_channel_group.queue_bytes) > 0:
                    trigger_reason = "initial"
                elif int(logical_channel_group.queue_bytes) == 0 and previous_bsr_reported_bytes != 0:
                    trigger_reason = "buffer_empty"
                elif int(logical_channel_group.queue_bytes) > previous_bsr_reported_bytes:
                    trigger_reason = "buffer_growth"
                elif periodic_timer_expired:
                    trigger_reason = "periodic"

            if trigger_reason is not None and current_slot_id is not None:
                logical_channel_group.bsr_reported_bytes = int(logical_channel_group.queue_bytes)
                logical_channel_group.last_bsr_report_slot = int(current_slot_id)
                logical_channel_group.last_bsr_trigger_slot = int(current_slot_id)
                logical_channel_group.bsr_triggered = True
                logical_channel_group.bsr_trigger_reason = str(trigger_reason)
            else:
                logical_channel_group.bsr_triggered = False
                if current_slot_id is None and previous_group is None and int(logical_channel_group.queue_bytes) > 0:
                    logical_channel_group.bsr_reported_bytes = int(logical_channel_group.queue_bytes)
                    logical_channel_group.bsr_triggered = True
                    logical_channel_group.bsr_trigger_reason = "initial"
                elif previous_group is None and int(logical_channel_group.queue_bytes) > 0:
                    logical_channel_group.bsr_reported_bytes = int(logical_channel_group.queue_bytes)
                    logical_channel_group.last_bsr_report_slot = current_slot_id
                    logical_channel_group.last_bsr_trigger_slot = current_slot_id
                    logical_channel_group.bsr_triggered = True
                    logical_channel_group.bsr_trigger_reason = "initial"
                elif previous_group is not None and int(logical_channel_group.queue_bytes) != previous_queue_bytes:
                    logical_channel_group.bsr_trigger_reason = "buffer_stale"

            logical_channel_group.bsr_pending_bytes = max(
                0,
                int(logical_channel_group.queue_bytes) - int(logical_channel_group.bsr_reported_bytes),
            )
            logical_channel_group.bsr_age_slots = (
                max(0, current_slot_id - int(logical_channel_group.last_bsr_report_slot))
                if current_slot_id is not None
                and logical_channel_group.last_bsr_report_slot is not None
                else getattr(logical_channel_group, "bsr_age_slots", None)
            )
        self._safe_set(ue, "logical_channel_groups", logical_channel_groups)
        return logical_channel_groups

    def refresh_logical_channel_groups(self, state: Any) -> None:
        current_slot_id = int(self._safe_get(state, "slot_id", 0))
        for ue in self._iter_ues(state):
            self._recompute_logical_channel_groups(ue, slot_id=current_slot_id)

    def _reset_logical_channel_slot_runtime(self, ue: Any, *, slot_id: int | None = None) -> None:
        for logical_channel in self._get_logical_channels(ue):
            try:
                logical_channel.scheduled_bytes = 0
                logical_channel.delivered_bytes = 0
                logical_channel.prioritized_scheduled_bytes = 0
            except Exception:
                pass
        self._recompute_logical_channel_groups(ue, slot_id=slot_id)

    def _recompute_ue_queue_bytes(self, ue: Any, *, slot_id: int | None = None) -> int:
        logical_channels = self._get_logical_channels(ue)
        if len(logical_channels) == 0:
            return int(self._safe_get(ue, "queue_bytes", 0))
        queue_bytes = int(
            sum(int(getattr(logical_channel, "queue_bytes", 0)) for logical_channel in logical_channels)
        )
        self._safe_set(ue, "queue_bytes", queue_bytes)
        self._recompute_logical_channel_groups(ue, slot_id=slot_id)
        return queue_bytes

    def _next_harq_redundancy_version(self, retransmission_round: int) -> int:
        normalized_round = max(0, int(retransmission_round))
        return int(self._HARQ_RV_SEQUENCE[min(normalized_round, len(self._HARQ_RV_SEQUENCE) - 1)])

    def _get_harq_processes(self, harq: Any) -> list[Any]:
        return list(getattr(harq, "processes", []) or [])

    def _set_harq_processes(self, harq: Any, processes: list[Any]) -> None:
        self._safe_set(harq, "processes", list(processes))

    def _get_harq_rv_sequence(self, process: Any | None) -> list[int]:
        sequence = list(getattr(process, "rv_sequence", []) or [])
        normalized_sequence = [int(item) for item in sequence if str(item).strip() != ""]
        if len(normalized_sequence) == 0:
            return list(self._HARQ_RV_SEQUENCE)
        return normalized_sequence

    def _clone_harq_process(
        self,
        process: Any | None,
        *,
        process_id: int,
        overrides: dict[str, Any] | None = None,
    ) -> HarqProcessState:
        base = process
        applied_overrides = dict(overrides or {})
        return HarqProcessState(
            process_id=int(process_id),
            is_busy=bool(applied_overrides.get("is_busy", getattr(base, "is_busy", False))),
            ndi=int(applied_overrides.get("ndi", getattr(base, "ndi", 0))),
            tb_id=applied_overrides.get("tb_id", getattr(base, "tb_id", None)),
            redundancy_version=int(
                applied_overrides.get("redundancy_version", getattr(base, "redundancy_version", 0))
            ),
            retransmission_round=int(
                applied_overrides.get(
                    "retransmission_round",
                    getattr(base, "retransmission_round", 0),
                )
            ),
            pending_bytes=int(applied_overrides.get("pending_bytes", getattr(base, "pending_bytes", 0))),
            last_tx_bytes=int(applied_overrides.get("last_tx_bytes", getattr(base, "last_tx_bytes", 0))),
            last_delivered_bytes=int(
                applied_overrides.get("last_delivered_bytes", getattr(base, "last_delivered_bytes", 0))
            ),
            is_waiting_ack=bool(
                applied_overrides.get("is_waiting_ack", getattr(base, "is_waiting_ack", False))
            ),
            rv_sequence=list(
                applied_overrides.get("rv_sequence", self._get_harq_rv_sequence(base))
            ),
            retx_deadline_slot=applied_overrides.get(
                "retx_deadline_slot",
                getattr(base, "retx_deadline_slot", None),
            ),
            timeout_slot=applied_overrides.get("timeout_slot", getattr(base, "timeout_slot", None)),
            flush_reason=applied_overrides.get("flush_reason", getattr(base, "flush_reason", None)),
            service_type=applied_overrides.get("service_type", getattr(base, "service_type", None)),
            max_retransmission_rounds=int(
                applied_overrides.get(
                    "max_retransmission_rounds",
                    getattr(base, "max_retransmission_rounds", 4),
                )
            ),
            reserved_at_slot=applied_overrides.get(
                "reserved_at_slot",
                getattr(base, "reserved_at_slot", None),
            ),
            last_tx_slot=applied_overrides.get("last_tx_slot", getattr(base, "last_tx_slot", None)),
            pending_logical_channels=list(
                applied_overrides.get(
                    "pending_logical_channels",
                    list(getattr(base, "pending_logical_channels", []) or []),
                )
            ),
        )

    def _record_harq_flush(self, harq: Any, *, reason: str, slot_id: int) -> None:
        self._safe_set(
            harq,
            "harq_flush_count",
            int(getattr(harq, "harq_flush_count", 0)) + 1,
        )
        self._safe_set(harq, "last_flush_reason", str(reason))
        if str(reason) in {"timeout", "retx_deadline"}:
            self._safe_set(harq, "last_timeout_slot", int(slot_id))

    def _synchronize_harq_legacy_fields(self, harq: Any) -> None:
        if harq is None:
            return
        processes = [
            process
            for process in self._get_harq_processes(harq)
            if int(getattr(process, "pending_bytes", 0)) > 0
        ]
        if len(processes) == 0:
            self._safe_set(harq, "pending_retx_bytes", 0)
            self._safe_set(harq, "pending_retx_logical_channels", [])
            self._safe_set(harq, "retransmission_count", 0)
            self._safe_set(harq, "is_waiting_ack", False)
            return

        active_process = sorted(
            processes,
            key=lambda process: (
                int(getattr(process, "retx_deadline_slot", 10**9) or 10**9),
                int(getattr(process, "timeout_slot", 10**9) or 10**9),
                -int(getattr(process, "retransmission_round", 0)),
                int(getattr(process, "process_id", 0)),
            ),
        )[0]
        self._safe_set(
            harq,
            "pending_retx_bytes",
            int(sum(int(getattr(process, "pending_bytes", 0)) for process in processes)),
        )
        self._safe_set(
            harq,
            "pending_retx_logical_channels",
            list(getattr(active_process, "pending_logical_channels", []) or []),
        )
        self._safe_set(harq, "process_id", int(getattr(active_process, "process_id", 0)))
        self._safe_set(harq, "ndi", int(getattr(active_process, "ndi", 0)))
        self._safe_set(
            harq,
            "redundancy_version",
            int(getattr(active_process, "redundancy_version", 0)),
        )
        self._safe_set(
            harq,
            "retransmission_count",
            int(getattr(active_process, "retransmission_round", 0)),
        )
        self._safe_set(
            harq,
            "last_harq_process_id",
            int(getattr(active_process, "process_id", 0)),
        )
        self._safe_set(
            harq,
            "last_retransmission_round",
            int(getattr(active_process, "retransmission_round", 0)),
        )
        self._safe_set(
            harq,
            "max_retransmission_rounds",
            int(getattr(active_process, "max_retransmission_rounds", 4)),
        )

    def _build_harq_pending_from_allocations(
        self,
        allocations: List[Any],
        *,
        process_id: int | None = None,
        redundancy_version: int = 0,
        ndi: int = 0,
        retransmission_round: int = 0,
    ) -> list[HarqLogicalChannelState]:
        pending_entries: list[HarqLogicalChannelState] = []
        for allocation in allocations:
            scheduled_bytes = int(getattr(allocation, "scheduled_bytes", 0))
            if scheduled_bytes <= 0:
                continue
            pending_entries.append(
                HarqLogicalChannelState(
                    logical_channel_id=str(getattr(allocation, "logical_channel_id", "")),
                    drb_id=(
                        int(getattr(allocation, "drb_id", 1))
                        if getattr(allocation, "drb_id", None) is not None
                        else None
                    ),
                    lcg_id=(
                        int(getattr(allocation, "lcg_id", 0))
                        if getattr(allocation, "lcg_id", None) is not None
                        else None
                    ),
                    pending_bytes=int(scheduled_bytes),
                    priority=int(getattr(allocation, "priority", 0)),
                    packet_delay_budget_ms=float(
                        getattr(allocation, "packet_delay_budget_ms", 0.0)
                    ),
                    slice_id=str(getattr(allocation, "slice_id", "")),
                    traffic_type=str(getattr(allocation, "traffic_type", "")),
                    process_id=(int(process_id) if process_id is not None else None),
                    redundancy_version=int(redundancy_version),
                    ndi=int(ndi),
                    retransmission_round=int(retransmission_round),
                )
            )
        return pending_entries

    def _apply_logical_channel_scheduled_bytes(
        self,
        ue: Any,
        allocations: List[Any],
        *,
        slot_id: int,
    ) -> None:
        for allocation in allocations:
            logical_channel_id = str(getattr(allocation, "logical_channel_id", ""))
            if logical_channel_id == "":
                continue
            logical_channel = self._get_logical_channel(ue, logical_channel_id)
            if logical_channel is None:
                continue
            try:
                logical_channel.scheduled_bytes = int(getattr(allocation, "scheduled_bytes", 0))
                logical_channel.prioritized_scheduled_bytes = int(
                    getattr(allocation, "prioritized_bytes", 0)
                )
                logical_channel.last_scheduled_slot = int(slot_id)
            except Exception:
                pass
        self._recompute_logical_channel_groups(ue, slot_id=slot_id)

    def _apply_logical_channel_delivery(
        self,
        ue: Any,
        allocations: List[Any],
        *,
        delivered_bytes: int,
        slot_id: int,
    ) -> None:
        remaining_bytes = max(0, int(delivered_bytes))
        if remaining_bytes <= 0:
            return
        for allocation in allocations:
            if remaining_bytes <= 0:
                break
            logical_channel_id = str(getattr(allocation, "logical_channel_id", ""))
            if logical_channel_id == "":
                continue
            scheduled_bytes = int(getattr(allocation, "scheduled_bytes", 0))
            delivered_for_channel = min(remaining_bytes, max(0, scheduled_bytes))
            if delivered_for_channel <= 0:
                continue
            logical_channel = self._get_logical_channel(ue, logical_channel_id)
            if logical_channel is None:
                continue
            current_queue_bytes = int(getattr(logical_channel, "queue_bytes", 0))
            new_queue_bytes = max(0, current_queue_bytes - delivered_for_channel)
            try:
                logical_channel.queue_bytes = int(new_queue_bytes)
                logical_channel.delivered_bytes = int(delivered_for_channel)
                logical_channel.last_delivered_slot = int(slot_id)
            except Exception:
                pass
            remaining_bytes -= int(delivered_for_channel)
        self._recompute_logical_channel_groups(ue, slot_id=slot_id)

    def _reduce_pending_harq_logical_channels(
        self,
        pending_entries: List[Any],
        *,
        delivered_bytes: int,
    ) -> list[HarqLogicalChannelState]:
        remaining_bytes = max(0, int(delivered_bytes))
        reduced_entries: list[HarqLogicalChannelState] = []
        for entry in pending_entries:
            current_pending_bytes = int(getattr(entry, "pending_bytes", 0))
            reduced_pending_bytes = current_pending_bytes
            if remaining_bytes > 0:
                served_bytes = min(remaining_bytes, current_pending_bytes)
                reduced_pending_bytes = max(0, current_pending_bytes - served_bytes)
                remaining_bytes -= int(served_bytes)
            if reduced_pending_bytes <= 0:
                continue
            reduced_entries.append(
                HarqLogicalChannelState(
                    logical_channel_id=str(getattr(entry, "logical_channel_id", "")),
                    drb_id=getattr(entry, "drb_id", None),
                    lcg_id=getattr(entry, "lcg_id", None),
                    pending_bytes=int(reduced_pending_bytes),
                    priority=int(getattr(entry, "priority", 0)),
                    packet_delay_budget_ms=float(
                        getattr(entry, "packet_delay_budget_ms", 0.0)
                    ),
                    slice_id=str(getattr(entry, "slice_id", "")),
                    traffic_type=str(getattr(entry, "traffic_type", "")),
                    process_id=getattr(entry, "process_id", None),
                    redundancy_version=int(getattr(entry, "redundancy_version", 0)),
                    ndi=int(getattr(entry, "ndi", 0)),
                    retransmission_round=int(getattr(entry, "retransmission_round", 0)),
                )
            )
        return reduced_entries

    def _build_ue_logical_channel_snapshot(self, ue: Any) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        for logical_channel in self._get_logical_channels(ue):
            snapshots.append(
                {
                    "logical_channel_id": str(getattr(logical_channel, "logical_channel_id", "")),
                    "drb_id": int(getattr(logical_channel, "drb_id", 1)),
                    "lcg_id": int(getattr(logical_channel, "lcg_id", 0)),
                    "qos_flow_id": getattr(logical_channel, "qos_flow_id", None),
                    "queue_bytes": int(getattr(logical_channel, "queue_bytes", 0)),
                    "priority": int(getattr(logical_channel, "priority", 0)),
                    "packet_delay_budget_ms": float(
                        getattr(logical_channel, "packet_delay_budget_ms", 0.0)
                    ),
                    "slice_id": str(getattr(logical_channel, "slice_id", "")),
                    "traffic_type": str(getattr(logical_channel, "traffic_type", "")),
                    "prioritized_bit_rate_bps": getattr(
                        logical_channel,
                        "prioritized_bit_rate_bps",
                        None,
                    ),
                    "bucket_size_duration_ms": float(
                        getattr(logical_channel, "bucket_size_duration_ms", 10.0)
                    ),
                    "pbr_credit_bytes": float(getattr(logical_channel, "pbr_credit_bytes", 0.0)),
                    "last_credit_update_slot": getattr(
                        logical_channel,
                        "last_credit_update_slot",
                        None,
                    ),
                    "scheduled_bytes": int(getattr(logical_channel, "scheduled_bytes", 0)),
                    "prioritized_scheduled_bytes": int(
                        getattr(logical_channel, "prioritized_scheduled_bytes", 0)
                    ),
                    "delivered_bytes": int(getattr(logical_channel, "delivered_bytes", 0)),
                    "last_scheduled_slot": getattr(logical_channel, "last_scheduled_slot", None),
                    "last_delivered_slot": getattr(logical_channel, "last_delivered_slot", None),
                }
            )
        return snapshots

    def _build_ue_logical_channel_group_snapshot(
        self,
        ue: Any,
        *,
        slot_id: int | None = None,
    ) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        for logical_channel_group in self._recompute_logical_channel_groups(ue, slot_id=slot_id):
            snapshots.append(
                {
                    "lcg_id": int(getattr(logical_channel_group, "lcg_id", 0)),
                    "logical_channel_ids": list(
                        getattr(logical_channel_group, "logical_channel_ids", []) or []
                    ),
                    "drb_ids": list(getattr(logical_channel_group, "drb_ids", []) or []),
                    "queue_bytes": int(getattr(logical_channel_group, "queue_bytes", 0)),
                    "prioritized_queue_bytes": int(
                        getattr(logical_channel_group, "prioritized_queue_bytes", 0)
                    ),
                    "pbr_credit_bytes": float(
                        getattr(logical_channel_group, "pbr_credit_bytes", 0.0)
                    ),
                    "scheduled_bytes": int(getattr(logical_channel_group, "scheduled_bytes", 0)),
                    "prioritized_scheduled_bytes": int(
                        getattr(logical_channel_group, "prioritized_scheduled_bytes", 0)
                    ),
                    "delivered_bytes": int(getattr(logical_channel_group, "delivered_bytes", 0)),
                    "bsr_reported_bytes": int(
                        getattr(logical_channel_group, "bsr_reported_bytes", 0)
                    ),
                    "bsr_pending_bytes": int(
                        getattr(logical_channel_group, "bsr_pending_bytes", 0)
                    ),
                    "bsr_age_slots": getattr(logical_channel_group, "bsr_age_slots", None),
                    "last_bsr_report_slot": getattr(
                        logical_channel_group, "last_bsr_report_slot", None
                    ),
                    "last_bsr_trigger_slot": getattr(
                        logical_channel_group, "last_bsr_trigger_slot", None
                    ),
                    "bsr_triggered": bool(
                        getattr(logical_channel_group, "bsr_triggered", False)
                    ),
                    "bsr_trigger_reason": getattr(
                        logical_channel_group,
                        "bsr_trigger_reason",
                        None,
                    ),
                }
            )
        return snapshots

    def _build_ue_true_snapshot(
        self,
        ue: Any,
        *,
        slot_id: int | None = None,
    ) -> Dict[str, Any]:
        return {
            "queue_bytes": int(self._safe_get(ue, "queue_bytes", 0)),
            "throughput_mbps": float(self._safe_get(ue, "throughput_mbps", 0.0)),
            "sinr_db": float(self._safe_get(ue, "sinr_db", 0.0)),
            "pre_eq_sinr_db": float(self._safe_get(ue, "pre_eq_sinr_db", 0.0)),
            "post_eq_sinr_db": float(self._safe_get(ue, "post_eq_sinr_db", 0.0)),
            "bler": float(self._safe_get(ue, "bler", 1.0)),
            "allocated_prbs": int(self._safe_get(ue, "allocated_prbs", 0)),
            "allocated_symbols": int(self._safe_get(ue, "allocated_symbols", 0)),
            "start_prb": int(self._safe_get(ue, "start_prb", 0)),
            "start_symbol": int(self._safe_get(ue, "start_symbol", 0)),
            "resource_block_indices": list(self._safe_get(ue, "resource_block_indices", []) or []),
            "allocated_symbol_indices": list(self._safe_get(ue, "allocated_symbol_indices", []) or []),
            "allocated_subcarrier_count": int(len(list(self._safe_get(ue, "allocated_subcarrier_indices", []) or []))),
            "is_retx": bool(self._safe_get(ue, "is_retx", False)),
            "channel_rank_limit": int(self._safe_get(ue, "channel_rank_limit", 1)),
            "precoder_match_score": float(self._safe_get(ue, "precoder_match_score", 1.0)),
            "mcs_index": int(self._safe_get(ue, "mcs_index", 0)),
            "grant_target_mcs": self._safe_get(ue, "grant_target_mcs", None),
            "grant_bytes_budget": int(self._safe_get(ue, "grant_bytes_budget", 0)),
            "grant_mcs_table_id": str(self._safe_get(ue, "grant_mcs_table_id", "qam64")),
            "grant_tbs_selection_method": str(
                self._safe_get(ue, "grant_tbs_selection_method", "resource_elements")
            ),
            "grant_dmrs_pattern_id": self._safe_get(ue, "grant_dmrs_pattern_id", None),
            "grant_cqi_validity_age_slots": int(
                self._safe_get(ue, "grant_cqi_validity_age_slots", 0)
            ),
            "grant_ri_validity_age_slots": int(
                self._safe_get(ue, "grant_ri_validity_age_slots", 0)
            ),
            "grant_pmi_validity_age_slots": int(
                self._safe_get(ue, "grant_pmi_validity_age_slots", 0)
            ),
            "grant_channel_quality_confidence": float(
                self._safe_get(ue, "grant_channel_quality_confidence", 1.0)
            ),
            "grant_link_adaptation_reason": str(
                self._safe_get(ue, "grant_link_adaptation_reason", "")
            ),
            "phy_mode": str(self._safe_get(ue, "phy_mode", "system_level_approx")),
            "waveform_backend": self._safe_get(ue, "waveform_backend", None),
            "effective_data_re_count": int(self._safe_get(ue, "effective_data_re_count", 0)),
            "dmrs_overhead_fraction": float(self._safe_get(ue, "dmrs_overhead_fraction", 0.0)),
            "scheduled_bytes": int(self._safe_get(ue, "scheduled_bytes", 0)),
            "delivered_bytes": int(self._safe_get(ue, "delivered_bytes", 0)),
            "new_tx_bytes": int(self._safe_get(ue, "new_tx_bytes", 0)),
            "retx_tx_bytes": int(self._safe_get(ue, "retx_tx_bytes", 0)),
            "delivered_new_bytes": int(self._safe_get(ue, "delivered_new_bytes", 0)),
            "delivered_retx_bytes": int(self._safe_get(ue, "delivered_retx_bytes", 0)),
            "is_scheduled": bool(self._safe_get(ue, "is_scheduled", False)),
            "resource_grid_re_count": int(self._safe_get(ue, "resource_grid_re_count", 0)),
            "resource_grid_overlap_fraction": float(self._safe_get(ue, "resource_grid_overlap_fraction", 0.0)),
            "interference_penalty_db": float(self._safe_get(ue, "interference_penalty_db", 0.0)),
            "intra_cell_interference_penalty_db": float(
                self._safe_get(ue, "intra_cell_interference_penalty_db", 0.0)
            ),
            "inter_cell_interference_dbm": float(
                self._safe_get(ue, "inter_cell_interference_dbm", -120.0)
            ),
            "inter_cell_interference_penalty_db": float(
                self._safe_get(ue, "inter_cell_interference_penalty_db", 0.0)
            ),
            "inter_cell_interference_overlap_fraction": float(
                self._safe_get(ue, "inter_cell_interference_overlap_fraction", 0.0)
            ),
            "inter_cell_frequency_coupling_fraction": float(
                self._safe_get(ue, "inter_cell_frequency_coupling_fraction", 0.0)
            ),
            "inter_cell_interference_source_count": int(
                self._safe_get(ue, "inter_cell_interference_source_count", 0)
            ),
            "spatial_group_id": self._safe_get(ue, "spatial_group_id", None),
            "spatial_group_size": int(self._safe_get(ue, "spatial_group_size", 1)),
            "spatial_compatibility_score": float(self._safe_get(ue, "spatial_compatibility_score", 0.0)),
            "spatial_multiplexing_gain_db": float(self._safe_get(ue, "spatial_multiplexing_gain_db", 0.0)),
            "spatial_mode": str(self._safe_get(ue, "spatial_mode", "su_siso")),
            "logical_channel_count": int(len(self._get_logical_channels(ue))),
            "logical_channel_group_count": int(
                len(self._recompute_logical_channel_groups(ue, slot_id=slot_id))
            ),
            "scheduled_logical_channel_ids": list(
                self._safe_get(ue, "scheduled_logical_channel_ids", []) or []
            ),
            "scheduled_drb_ids": list(self._safe_get(ue, "scheduled_drb_ids", []) or []),
            "scheduled_lcg_ids": list(self._safe_get(ue, "scheduled_lcg_ids", []) or []),
            "grant_selection_reason": str(self._safe_get(ue, "grant_selection_reason", "")),
            "grant_rejection_reason": str(self._safe_get(ue, "grant_rejection_reason", "")),
            "grant_resource_pressure_source": str(
                self._safe_get(ue, "grant_resource_pressure_source", "")
            ),
            "grant_harq_preemption_cause": self._safe_get(
                ue,
                "grant_harq_preemption_cause",
                None,
            ),
            "grant_constraint_hits": list(self._safe_get(ue, "grant_constraint_hits", []) or []),
            "grant_lc_multiplexing_summary": dict(
                self._safe_get(ue, "grant_lc_multiplexing_summary", {}) or {}
            ),
            "harq_process_id": self._safe_get(self._safe_get(ue, "harq", None), "process_id", None),
            "harq_redundancy_version": self._safe_get(
                self._safe_get(ue, "harq", None),
                "redundancy_version",
                None,
            ),
            "harq_ndi": self._safe_get(self._safe_get(ue, "harq", None), "ndi", None),
            "harq_retransmission_round": int(
                self._safe_get(self._safe_get(ue, "harq", None), "retransmission_count", 0)
            ),
            "harq_tb_id": self._safe_get(
                self._get_harq_processes(self._safe_get(ue, "harq", None))[0]
                if len(self._get_harq_processes(self._safe_get(ue, "harq", None))) > 0
                else None,
                "tb_id",
                None,
            ),
            "harq_retx_deadline_slot": self._safe_get(
                self._get_harq_processes(self._safe_get(ue, "harq", None))[0]
                if len(self._get_harq_processes(self._safe_get(ue, "harq", None))) > 0
                else None,
                "retx_deadline_slot",
                None,
            ),
            "harq_timeout_slot": self._safe_get(
                self._get_harq_processes(self._safe_get(ue, "harq", None))[0]
                if len(self._get_harq_processes(self._safe_get(ue, "harq", None))) > 0
                else None,
                "timeout_slot",
                None,
            ),
            "harq_last_flush_reason": self._safe_get(
                self._safe_get(ue, "harq", None),
                "last_flush_reason",
                None,
            ),
            "harq_flush_count": int(
                self._safe_get(self._safe_get(ue, "harq", None), "harq_flush_count", 0)
            ),
            "harq_max_retransmission_rounds": int(
                self._safe_get(self._safe_get(ue, "harq", None), "max_retransmission_rounds", 0)
            ),
            "active_harq_process_count": int(
                len(self._get_harq_processes(self._safe_get(ue, "harq", None)))
            ) if self._safe_get(ue, "harq", None) is not None else 0,
        }

    def _build_ue_connection_snapshot(self, connection: Any | None, ue: Any) -> Dict[str, Any]:
        default_serving_beam_id = int(self._safe_get(ue, "serving_beam_id", 0))
        if connection is None:
            return {
                "serving_cell_id": None,
                "candidate_cells": [],
                "connection_state": "connected",
                "c_rnti": None,
                "last_scheduled_slot": None,
                "last_ack_slot": None,
                "serving_beam_id": default_serving_beam_id,
                "rank_indicator": int(self._safe_get(ue, "rank", 1)),
                "pmi": int(self._safe_get(ue, "pmi", 0)),
                "cqi": int(self._safe_get(ue, "cqi", 0)),
                "active_harq_processes": [],
                "preferred_target_cell_id": None,
                "handover_pending_target_cell_id": None,
                "handover_started_slot": None,
                "last_handover_slot": None,
                "handover_count": 0,
                "handover_cost_db": 0.0,
                "last_handover_cost_db": 0.0,
                "handover_interruption_slots": 0,
                "interruption_remaining_slots": 0,
                "steering_hysteresis_db": 0.0,
                "measurement_event_trigger": "a3_offset",
                "measurement_event_offset_db": 0.0,
                "last_measurement_event_slot": None,
                "last_measurement_event_type": None,
                "last_measurement_event_margin_db": None,
                "last_trigger_reason": None,
                "traffic_steering_bias_db": 0.0,
                "consecutive_nack_count": 0,
                "out_of_sync_reason": None,
            }

        return {
            "serving_cell_id": getattr(connection, "serving_cell_id", None),
            "candidate_cells": [
                {
                    "cell_id": str(getattr(candidate, "cell_id", "")),
                    "cell_type": str(getattr(candidate, "cell_type", "macro")),
                    "distance_2d_m": float(getattr(candidate, "distance_2d_m", 0.0)),
                    "carrier_frequency_hz": float(getattr(candidate, "carrier_frequency_hz", 3.5e9)),
                    "bandwidth_hz": float(getattr(candidate, "bandwidth_hz", 20e6)),
                    "subcarrier_spacing_hz": float(getattr(candidate, "subcarrier_spacing_hz", 30e3)),
                    "tx_power_dbm": float(getattr(candidate, "tx_power_dbm", 30.0)),
                    "connected_ue_count": int(getattr(candidate, "connected_ue_count", 0)),
                    "scheduled_ue_count": int(getattr(candidate, "scheduled_ue_count", 0)),
                    "queued_bytes": int(getattr(candidate, "queued_bytes", 0)),
                    "estimated_load": float(getattr(candidate, "estimated_load", 0.0)),
                    "load_penalty_db": float(getattr(candidate, "load_penalty_db", 0.0)),
                    "estimated_interference_db": float(getattr(candidate, "estimated_interference_db", -120.0)),
                    "interference_penalty_db": float(getattr(candidate, "interference_penalty_db", 0.0)),
                    "signal_margin_db": float(getattr(candidate, "signal_margin_db", -120.0)),
                    "handover_cost_db": float(getattr(candidate, "handover_cost_db", 0.0)),
                    "steering_hysteresis_db": float(getattr(candidate, "steering_hysteresis_db", 0.0)),
                    "hysteresis_adjusted_score_db": float(
                        getattr(candidate, "hysteresis_adjusted_score_db", -100.0)
                    ),
                    "measurement_event_trigger": str(
                        getattr(candidate, "measurement_event_trigger", "a3_offset")
                    ),
                    "measurement_event_offset_db": float(
                        getattr(candidate, "measurement_event_offset_db", 0.0)
                    ),
                    "measurement_event_met": bool(getattr(candidate, "measurement_event_met", False)),
                    "trigger_margin_db": float(getattr(candidate, "trigger_margin_db", -100.0)),
                    "steering_score_db": float(getattr(candidate, "steering_score_db", -100.0)),
                    "pathloss_db": float(getattr(candidate, "pathloss_db", 130.0)),
                    "rsrp_dbm": float(getattr(candidate, "rsrp_dbm", -100.0)),
                    "quality_score_db": float(getattr(candidate, "quality_score_db", -100.0)),
                    "candidate_beam_id": int(getattr(candidate, "candidate_beam_id", 0)),
                    "is_serving": bool(getattr(candidate, "is_serving", False)),
                    "is_target": bool(getattr(candidate, "is_target", False)),
                }
                for candidate in list(getattr(connection, "candidate_cells", []) or [])
            ],
            "connection_state": str(getattr(connection, "connection_state", "connected")),
            "c_rnti": getattr(connection, "c_rnti", None),
            "last_scheduled_slot": getattr(connection, "last_scheduled_slot", None),
            "last_ack_slot": getattr(connection, "last_ack_slot", None),
            "serving_beam_id": int(getattr(connection, "serving_beam_id", default_serving_beam_id)),
            "rank_indicator": int(getattr(connection, "rank_indicator", self._safe_get(ue, "rank", 1))),
            "pmi": int(getattr(connection, "pmi", self._safe_get(ue, "pmi", 0))),
            "cqi": int(getattr(connection, "cqi", self._safe_get(ue, "cqi", 0))),
            "active_harq_processes": list(getattr(connection, "active_harq_processes", [])),
            "preferred_target_cell_id": getattr(connection, "preferred_target_cell_id", None),
            "handover_pending_target_cell_id": getattr(connection, "handover_pending_target_cell_id", None),
            "handover_started_slot": getattr(connection, "handover_started_slot", None),
            "last_handover_slot": getattr(connection, "last_handover_slot", None),
            "handover_count": int(getattr(connection, "handover_count", 0)),
            "handover_cost_db": float(getattr(connection, "handover_cost_db", 0.0)),
            "last_handover_cost_db": float(getattr(connection, "last_handover_cost_db", 0.0)),
            "handover_interruption_slots": int(getattr(connection, "handover_interruption_slots", 0)),
            "interruption_remaining_slots": int(getattr(connection, "interruption_remaining_slots", 0)),
            "steering_hysteresis_db": float(getattr(connection, "steering_hysteresis_db", 0.0)),
            "measurement_event_trigger": str(getattr(connection, "measurement_event_trigger", "a3_offset")),
            "measurement_event_offset_db": float(getattr(connection, "measurement_event_offset_db", 0.0)),
            "last_measurement_event_slot": getattr(connection, "last_measurement_event_slot", None),
            "last_measurement_event_type": getattr(connection, "last_measurement_event_type", None),
            "last_measurement_event_margin_db": getattr(connection, "last_measurement_event_margin_db", None),
            "last_trigger_reason": getattr(connection, "last_trigger_reason", None),
            "traffic_steering_bias_db": float(getattr(connection, "traffic_steering_bias_db", 0.0)),
            "consecutive_nack_count": int(getattr(connection, "consecutive_nack_count", 0)),
            "out_of_sync_reason": getattr(connection, "out_of_sync_reason", None),
        }

    def _build_ue_link_snapshot(self, link: Any | None, state: Any) -> Dict[str, Any]:
        default_channel = getattr(state, "channel", None)
        if link is None:
            return {
                "serving_cell_id": None,
                "scenario_type": None,
                "los_state": None,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.5,
                "distance_2d_m": 0.0,
                "distance_3d_m": 0.0,
                "velocity_mps": 0.0,
                "heading_deg": 0.0,
                "is_indoor": False,
                "pathloss_db": float(getattr(default_channel, "pathloss_db", 0.0)),
                "shadowing_db": float(getattr(default_channel, "shadowing_db", 0.0)),
                "o2i_penetration_loss_db": 0.0,
                "doppler_hz": float(getattr(default_channel, "doppler_hz", 0.0)),
                "delay_spread_ns": 300.0,
                "k_factor_db": 0.0,
                "angle_spread_deg": 35.0,
                "spatial_correlation_level": 0.5,
                "channel_model_type": str(getattr(default_channel, "channel_profile", "tdl_c")),
                "fast_fading_seed": 0,
                "spatial_corr_seed": 0,
                "last_pre_eq_sinr_db": None,
                "last_post_eq_sinr_db": None,
                "cqi": 0,
                "ri": 1,
                "pmi": 0,
                "large_scale": {
                    "distance_2d_m": 0.0,
                    "distance_3d_m": 0.0,
                    "pathloss_db": float(getattr(default_channel, "pathloss_db", 0.0)),
                    "shadowing_db": float(getattr(default_channel, "shadowing_db", 0.0)),
                    "los_state": None,
                    "o2i_penetration_loss_db": 0.0,
                },
                "small_scale": {
                    "doppler_hz": float(getattr(default_channel, "doppler_hz", 0.0)),
                    "delay_spread_ns": 300.0,
                    "k_factor_db": 0.0,
                    "angle_spread_deg": 35.0,
                    "spatial_correlation_level": 0.5,
                    "channel_model_type": str(getattr(default_channel, "channel_profile", "tdl_c")),
                    "fast_fading_seed": 0,
                    "spatial_corr_seed": 0,
                },
            }

        large_scale = getattr(link, "large_scale", None)
        small_scale = getattr(link, "small_scale", None)
        return {
            "serving_cell_id": getattr(link, "serving_cell_id", None),
            "scenario_type": str(getattr(link, "scenario_type", "")),
            "los_state": str(getattr(link, "los_state", "")),
            "x_m": float(getattr(link, "x_m", 0.0)),
            "y_m": float(getattr(link, "y_m", 0.0)),
            "z_m": float(getattr(link, "z_m", 1.5)),
            "distance_2d_m": float(getattr(link, "distance_2d_m", 0.0)),
            "distance_3d_m": float(getattr(link, "distance_3d_m", 0.0)),
            "velocity_mps": float(getattr(link, "velocity_mps", 0.0)),
            "heading_deg": float(getattr(link, "heading_deg", 0.0)),
            "is_indoor": bool(getattr(link, "is_indoor", False)),
            "pathloss_db": float(getattr(link, "pathloss_db", 0.0)),
            "shadowing_db": float(getattr(link, "shadowing_db", 0.0)),
            "o2i_penetration_loss_db": float(getattr(link, "o2i_penetration_loss_db", 0.0)),
            "doppler_hz": float(getattr(link, "doppler_hz", 0.0)),
            "delay_spread_ns": float(getattr(link, "delay_spread_ns", 300.0)),
            "k_factor_db": float(getattr(link, "k_factor_db", 0.0)),
            "angle_spread_deg": float(getattr(link, "angle_spread_deg", 35.0)),
            "spatial_correlation_level": float(getattr(link, "spatial_correlation_level", 0.5)),
            "channel_model_type": str(getattr(link, "channel_model_type", "tdl_c")),
            "fast_fading_seed": int(getattr(link, "fast_fading_seed", 0)),
            "spatial_corr_seed": int(getattr(link, "spatial_corr_seed", 0)),
            "last_pre_eq_sinr_db": getattr(link, "last_pre_eq_sinr_db", None),
            "last_post_eq_sinr_db": getattr(link, "last_post_eq_sinr_db", None),
            "cqi": int(getattr(link, "cqi", 0)),
            "ri": int(getattr(link, "ri", 1)),
            "pmi": int(getattr(link, "pmi", 0)),
            "large_scale": {
                "distance_2d_m": float(
                    getattr(large_scale, "distance_2d_m", getattr(link, "distance_2d_m", 0.0))
                ),
                "distance_3d_m": float(
                    getattr(large_scale, "distance_3d_m", getattr(link, "distance_3d_m", 0.0))
                ),
                "pathloss_db": float(
                    getattr(large_scale, "pathloss_db", getattr(link, "pathloss_db", 0.0))
                ),
                "shadowing_db": float(
                    getattr(large_scale, "shadowing_db", getattr(link, "shadowing_db", 0.0))
                ),
                "los_state": str(
                    getattr(large_scale, "los_state", getattr(link, "los_state", ""))
                ),
                "o2i_penetration_loss_db": float(
                    getattr(
                        large_scale,
                        "o2i_penetration_loss_db",
                        getattr(link, "o2i_penetration_loss_db", 0.0),
                    )
                ),
            },
            "small_scale": {
                "doppler_hz": float(
                    getattr(small_scale, "doppler_hz", getattr(link, "doppler_hz", 0.0))
                ),
                "delay_spread_ns": float(
                    getattr(small_scale, "delay_spread_ns", getattr(link, "delay_spread_ns", 300.0))
                ),
                "k_factor_db": float(
                    getattr(small_scale, "k_factor_db", getattr(link, "k_factor_db", 0.0))
                ),
                "angle_spread_deg": float(
                    getattr(
                        small_scale,
                        "angle_spread_deg",
                        getattr(link, "angle_spread_deg", 35.0),
                    )
                ),
                "spatial_correlation_level": float(
                    getattr(
                        small_scale,
                        "spatial_correlation_level",
                        getattr(link, "spatial_correlation_level", 0.5),
                    )
                ),
                "channel_model_type": str(
                    getattr(
                        small_scale,
                        "channel_model_type",
                        getattr(link, "channel_model_type", "tdl_c"),
                    )
                ),
                "fast_fading_seed": int(
                    getattr(small_scale, "fast_fading_seed", getattr(link, "fast_fading_seed", 0))
                ),
                "spatial_corr_seed": int(
                    getattr(small_scale, "spatial_corr_seed", getattr(link, "spatial_corr_seed", 0))
                ),
            },
        }

    def _build_ue_beamforming_snapshot(self, beamforming: Any | None, ue: Any, state: Any) -> Dict[str, Any]:
        antenna_state = getattr(state, "antenna", None)
        ue_id = str(self._safe_get(ue, "ue_id", "unknown_ue"))
        ue_num_rx_ant_by_ue = getattr(antenna_state, "ue_num_rx_ant_by_ue", {}) if antenna_state is not None else {}
        num_rx_ant = ue_num_rx_ant_by_ue.get(ue_id, 1) if isinstance(ue_num_rx_ant_by_ue, dict) else 1
        if beamforming is None:
            return {
                "codebook_id": int(getattr(getattr(state, "cell", None), "current_codebook_id", 0)),
                "beam_profile_id": int(getattr(getattr(state, "cell", None), "current_beam_profile_id", 0)),
                "serving_beam_id": int(self._safe_get(ue, "serving_beam_id", 0)),
                "precoder_id": int(getattr(getattr(state, "cell", None), "current_codebook_id", 0)),
                "combiner_id": int(self._safe_get(ue, "pmi", 0)),
                "rank": int(self._safe_get(ue, "rank", 1)),
                "layer_count": int(self._safe_get(ue, "ri", 1)),
                "beam_gain_db": 0.0,
                "beam_direction_deg": float(int(self._safe_get(ue, "serving_beam_id", 0)) * 30.0),
                "precoding_matrix_label": None,
                "precoder_matrix": [],
                "combiner_vector": [],
                "precoder_match_score": 1.0,
                "post_eq_sinr_db": None,
                "spatial_group_id": None,
                "num_tx_ant": int(getattr(antenna_state, "gnb_num_tx_ant", 1)) if antenna_state is not None else 1,
                "num_rx_ant": int(num_rx_ant),
            }

        return {
            "codebook_id": int(getattr(beamforming, "codebook_id", 0)),
            "beam_profile_id": int(getattr(beamforming, "beam_profile_id", 0)),
            "serving_beam_id": int(getattr(beamforming, "serving_beam_id", self._safe_get(ue, "serving_beam_id", 0))),
            "precoder_id": int(getattr(beamforming, "precoder_id", 0)),
            "combiner_id": int(getattr(beamforming, "combiner_id", self._safe_get(ue, "pmi", 0))),
            "rank": int(getattr(beamforming, "rank", self._safe_get(ue, "rank", 1))),
            "layer_count": int(getattr(beamforming, "layer_count", self._safe_get(ue, "ri", 1))),
            "beam_gain_db": float(getattr(beamforming, "beam_gain_db", 0.0)),
            "beam_direction_deg": float(getattr(beamforming, "beam_direction_deg", 0.0)),
            "precoding_matrix_label": getattr(beamforming, "precoding_matrix_label", None),
            "precoder_matrix": list(getattr(beamforming, "precoder_matrix", []) or []),
            "combiner_vector": list(getattr(beamforming, "combiner_vector", []) or []),
            "precoder_match_score": float(getattr(beamforming, "precoder_match_score", 1.0)),
            "post_eq_sinr_db": getattr(beamforming, "post_eq_sinr_db", None),
            "spatial_group_id": getattr(beamforming, "spatial_group_id", None),
            "num_tx_ant": int(getattr(antenna_state, "gnb_num_tx_ant", 1)) if antenna_state is not None else 1,
            "num_rx_ant": int(num_rx_ant),
        }

    def _build_antenna_snapshot(self, state: Any) -> Dict[str, Any]:
        antenna_state = getattr(state, "antenna", None)
        if antenna_state is None:
            return {
                "gnb_num_tx_ant": 1,
                "carrier_frequency_hz": None,
                "bandwidth_hz": None,
                "subcarrier_spacing_hz": None,
                "cyclic_prefix_type": None,
                "num_ofdm_symbols_per_slot": None,
                "gnb_panel_layout": None,
                "polarization": None,
            }

        return {
            "gnb_num_tx_ant": int(getattr(antenna_state, "gnb_num_tx_ant", 1)),
            "carrier_frequency_hz": float(getattr(antenna_state, "carrier_frequency_hz", 0.0)),
            "bandwidth_hz": float(getattr(antenna_state, "bandwidth_hz", 0.0)),
            "subcarrier_spacing_hz": float(getattr(antenna_state, "subcarrier_spacing_hz", 0.0)),
            "cyclic_prefix_type": getattr(antenna_state, "cyclic_prefix_type", None),
            "num_ofdm_symbols_per_slot": int(getattr(antenna_state, "num_ofdm_symbols_per_slot", 0)),
            "gnb_panel_layout": getattr(antenna_state, "gnb_panel_layout", None),
            "polarization": getattr(antenna_state, "polarization", None),
        }

    def _compute_grant_overlap_pair_count(self, grants: List[Any]) -> int:
        overlap_pair_count = 0
        for index, left_grant in enumerate(grants):
            left_start_prb = int(getattr(left_grant, "start_prb", 0))
            left_end_prb = left_start_prb + int(getattr(left_grant, "allocated_prbs", 0))
            left_start_symbol = int(getattr(left_grant, "start_symbol", 0))
            left_end_symbol = left_start_symbol + int(getattr(left_grant, "allocated_symbols", 0))
            for right_grant in grants[index + 1:]:
                right_start_prb = int(getattr(right_grant, "start_prb", 0))
                right_end_prb = right_start_prb + int(getattr(right_grant, "allocated_prbs", 0))
                right_start_symbol = int(getattr(right_grant, "start_symbol", 0))
                right_end_symbol = right_start_symbol + int(getattr(right_grant, "allocated_symbols", 0))
                overlap_prbs = max(0, min(left_end_prb, right_end_prb) - max(left_start_prb, right_start_prb))
                overlap_symbols = max(
                    0,
                    min(left_end_symbol, right_end_symbol) - max(left_start_symbol, right_start_symbol),
                )
                if overlap_prbs > 0 and overlap_symbols > 0:
                    overlap_pair_count += 1
        return overlap_pair_count

    def _interval_overlap_count(self, left_start: int, left_size: int, right_start: int, right_size: int) -> int:
        left_end = int(left_start) + max(0, int(left_size))
        right_end = int(right_start) + max(0, int(right_size))
        return max(0, min(left_end, right_end) - max(int(left_start), int(right_start)))

    def _intervals_are_adjacent(self, left_start: int, left_size: int, right_start: int, right_size: int) -> bool:
        overlap = self._interval_overlap_count(left_start, left_size, right_start, right_size)
        if overlap > 0:
            return False

        left_end = int(left_start) + max(0, int(left_size))
        right_end = int(right_start) + max(0, int(right_size))
        return left_end == int(right_start) or right_end == int(left_start)

    def _build_interference_groups(
        self,
        allocations: list[ResourceAllocationState],
    ) -> tuple[int, list[list[str]]]:
        adjacent_interference_pair_count = 0
        graph: dict[str, set[str]] = {str(item.ue_id): set() for item in allocations}

        for index, left in enumerate(allocations):
            for right in allocations[index + 1:]:
                symbol_overlap = self._interval_overlap_count(
                    int(getattr(left, "start_symbol", 0)),
                    int(getattr(left, "allocated_symbols", 0)),
                    int(getattr(right, "start_symbol", 0)),
                    int(getattr(right, "allocated_symbols", 0)),
                )
                if symbol_overlap <= 0:
                    continue

                prb_overlap = self._interval_overlap_count(
                    int(getattr(left, "start_prb", 0)),
                    int(getattr(left, "allocated_prbs", 0)),
                    int(getattr(right, "start_prb", 0)),
                    int(getattr(right, "allocated_prbs", 0)),
                )
                prb_adjacent = self._intervals_are_adjacent(
                    int(getattr(left, "start_prb", 0)),
                    int(getattr(left, "allocated_prbs", 0)),
                    int(getattr(right, "start_prb", 0)),
                    int(getattr(right, "allocated_prbs", 0)),
                )

                if prb_overlap <= 0 and not prb_adjacent:
                    continue

                left_ue_id = str(getattr(left, "ue_id", "unknown_ue"))
                right_ue_id = str(getattr(right, "ue_id", "unknown_ue"))
                graph[left_ue_id].add(right_ue_id)
                graph[right_ue_id].add(left_ue_id)
                if prb_overlap <= 0 and prb_adjacent:
                    adjacent_interference_pair_count += 1

        interference_groups: list[list[str]] = []
        visited: set[str] = set()
        next_group_id = 0
        allocation_map = {str(item.ue_id): item for item in allocations}

        for ue_id, neighbors in graph.items():
            if ue_id in visited or len(neighbors) == 0:
                continue

            stack = [ue_id]
            component: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in graph.get(current, set()):
                    if neighbor not in visited:
                        stack.append(neighbor)

            ordered_component = sorted(component)
            interference_groups.append(ordered_component)
            for component_ue_id in ordered_component:
                allocation = allocation_map.get(component_ue_id)
                if allocation is not None:
                    allocation.interference_group_id = next_group_id
            next_group_id += 1

        for allocation in allocations:
            if str(getattr(allocation, "ue_id", "")) not in visited:
                allocation.interference_group_id = None

        return adjacent_interference_pair_count, interference_groups

    def _build_spatial_groups(self, allocations: list[ResourceAllocationState]) -> list[list[str]]:
        grouped: dict[int, list[str]] = {}
        for allocation in allocations:
            spatial_group_id = getattr(allocation, "spatial_group_id", None)
            if spatial_group_id is None:
                continue
            grouped.setdefault(int(spatial_group_id), []).append(str(getattr(allocation, "ue_id", "unknown_ue")))

        spatial_groups: list[list[str]] = []
        for spatial_group_id in sorted(grouped):
            members = sorted(grouped[spatial_group_id])
            if len(members) > 1:
                spatial_groups.append(members)
        return spatial_groups

    def _update_resource_grid_state(self, state: Any, phy_result: Any | None) -> None:
        scheduler_state = getattr(state, "scheduler", None)
        scheduling_decision = getattr(scheduler_state, "last_decision", None) if scheduler_state is not None else None
        grants = list(getattr(scheduler_state, "grants", [])) if scheduler_state is not None else []
        total_prbs = int(getattr(scheduling_decision, "total_prbs", 0))
        total_symbols = int(getattr(scheduling_decision, "total_symbols", 0))
        existing_resource_grid = getattr(state, "resource_grid", None)
        subcarriers_per_prb = int(getattr(existing_resource_grid, "subcarriers_per_prb", 12))

        ue_results = list(getattr(phy_result, "ue_results", [])) if phy_result is not None else []
        ue_result_map = {
            str(getattr(item, "ue_id", "")): item
            for item in ue_results
            if str(getattr(item, "ue_id", "")) != ""
        }
        allocations: list[ResourceAllocationState] = []
        for grant in grants:
            ue_id = str(getattr(grant, "ue_id", "unknown_ue"))
            ue = self._find_ue_by_id(state, ue_id)
            ue_result = ue_result_map.get(ue_id)
            layer_count = int(
                self._safe_get(
                    ue_result,
                    "num_layers",
                    self._safe_get(ue, "ri", 1) if ue is not None else 1,
                )
            )
            resource_elements = int(
                self._safe_get(
                    ue_result,
                    "resource_grid_re_count",
                    int(getattr(grant, "allocated_prbs", 0))
                    * int(getattr(grant, "allocated_symbols", 0))
                    * subcarriers_per_prb,
                )
            )
            overlap_fraction = float(self._safe_get(ue_result, "resource_grid_overlap_fraction", 0.0))
            interference_penalty_db = float(self._safe_get(ue_result, "interference_penalty_db", 0.0))
            resource_block_indices = list(getattr(grant, "resource_block_indices", []) or [])
            allocated_symbol_indices = list(getattr(grant, "ofdm_symbol_indices", []) or [])
            allocated_subcarrier_indices = list(
                self._safe_get(ue, "allocated_subcarrier_indices", []) if ue is not None else []
            ) or []
            if not resource_block_indices:
                start_prb = int(getattr(grant, "start_prb", 0))
                allocated_prbs = int(getattr(grant, "allocated_prbs", 0))
                resource_block_indices = list(range(start_prb, start_prb + max(0, allocated_prbs)))
            if not allocated_symbol_indices:
                start_symbol = int(getattr(grant, "start_symbol", 0))
                allocated_symbols = int(getattr(grant, "allocated_symbols", 0))
                allocated_symbol_indices = list(range(start_symbol, start_symbol + max(0, allocated_symbols)))
            if not allocated_subcarrier_indices:
                for prb_index in resource_block_indices:
                    base_subcarrier = int(prb_index) * subcarriers_per_prb
                    allocated_subcarrier_indices.extend(
                        range(base_subcarrier, base_subcarrier + subcarriers_per_prb)
                    )
            allocations.append(
                ResourceAllocationState(
                    ue_id=ue_id,
                    start_prb=int(getattr(grant, "start_prb", 0)),
                    allocated_prbs=int(getattr(grant, "allocated_prbs", 0)),
                    start_symbol=int(getattr(grant, "start_symbol", 0)),
                    allocated_symbols=int(getattr(grant, "allocated_symbols", 0)),
                    resource_block_indices=resource_block_indices,
                    allocated_subcarrier_indices=allocated_subcarrier_indices,
                    allocated_symbol_indices=allocated_symbol_indices,
                    spatial_group_id=getattr(grant, "spatial_group_id", None),
                    layer_count=max(1, layer_count),
                    resource_elements=max(0, resource_elements),
                    overlap_fraction=max(0.0, overlap_fraction),
                    interference_penalty_db=max(0.0, interference_penalty_db),
                )
            )

        prb_symbol_occupancy_map = [
            [0 for _ in range(max(0, total_symbols))]
            for _ in range(max(0, total_prbs))
        ]
        ue_resource_slices: dict[str, ResourceAllocationState] = {}
        for allocation in allocations:
            ue_resource_slices[str(getattr(allocation, "ue_id", "unknown_ue"))] = allocation
            resource_block_indices = list(getattr(allocation, "resource_block_indices", []) or [])
            allocated_symbol_indices = list(getattr(allocation, "allocated_symbol_indices", []) or [])
            for prb_index in resource_block_indices:
                if prb_index < 0 or prb_index >= len(prb_symbol_occupancy_map):
                    continue
                row = prb_symbol_occupancy_map[prb_index]
                for symbol_index in allocated_symbol_indices:
                    if symbol_index < 0 or symbol_index >= len(row):
                        continue
                    row[symbol_index] += 1

        total_resource_elements = int(max(0, total_prbs) * max(0, total_symbols) * subcarriers_per_prb)
        allocated_resource_elements = int(sum(item.resource_elements for item in allocations))
        utilization = (
            float(allocated_resource_elements) / float(total_resource_elements)
            if total_resource_elements > 0
            else 0.0
        )
        overlap_values = [float(item.overlap_fraction) for item in allocations]
        interference_values = [float(item.interference_penalty_db) for item in allocations]
        adjacent_interference_pair_count, interference_groups = self._build_interference_groups(allocations)
        spatial_groups = self._build_spatial_groups(allocations)
        state.resource_grid = ResourceGridState(
            total_prbs=max(0, total_prbs),
            total_symbols=max(0, total_symbols),
            subcarriers_per_prb=subcarriers_per_prb,
            n_prbs_total=int(getattr(existing_resource_grid, "n_prbs_total", max(0, total_prbs))),
            n_subcarriers_per_prb=int(
                getattr(existing_resource_grid, "n_subcarriers_per_prb", subcarriers_per_prb)
            ),
            n_ofdm_symbols_per_slot=int(
                getattr(existing_resource_grid, "n_ofdm_symbols_per_slot", max(0, total_symbols))
            ),
            fft_size=int(getattr(existing_resource_grid, "fft_size", 0)),
            usable_subcarrier_mask=list(
                getattr(existing_resource_grid, "usable_subcarrier_mask", [])
            ),
            carrier_frequency_hz=float(
                getattr(existing_resource_grid, "carrier_frequency_hz", 3.5e9)
            ),
            bandwidth_hz=float(getattr(existing_resource_grid, "bandwidth_hz", 20e6)),
            subcarrier_spacing_hz=float(
                getattr(existing_resource_grid, "subcarrier_spacing_hz", 30e3)
            ),
            cyclic_prefix_type=str(
                getattr(existing_resource_grid, "cyclic_prefix_type", "normal")
            ),
            total_resource_elements=total_resource_elements,
            allocated_resource_elements=allocated_resource_elements,
            utilization=max(0.0, utilization),
            overlap_fraction_mean=float(self._mean_or_none(overlap_values) or 0.0),
            interference_penalty_db_mean=float(self._mean_or_none(interference_values) or 0.0),
            scheduled_ue_count=len(allocations),
            scheduled_layer_count_total=int(sum(item.layer_count for item in allocations)),
            mu_mimo_overlap_pair_count=self._compute_grant_overlap_pair_count(grants),
            mu_mimo_group_count=len(spatial_groups),
            mu_mimo_ue_count=sum(len(group) for group in spatial_groups),
            adjacent_interference_pair_count=int(adjacent_interference_pair_count),
            interference_groups=interference_groups,
            spatial_groups=spatial_groups,
            prb_symbol_occupancy_map=prb_symbol_occupancy_map,
            ue_resource_slices=ue_resource_slices,
            allocations=allocations,
        )

    def _build_resource_grid_snapshot(self, state: Any) -> Dict[str, Any]:
        resource_grid = getattr(state, "resource_grid", None)
        if resource_grid is None:
            return {
                "total_prbs": 0,
                "total_symbols": 0,
                "subcarriers_per_prb": 12,
                "n_prbs_total": 0,
                "n_subcarriers_per_prb": 12,
                "n_ofdm_symbols_per_slot": 0,
                "fft_size": 0,
                "usable_subcarrier_count": 0,
                "carrier_frequency_hz": 0.0,
                "bandwidth_hz": 0.0,
                "subcarrier_spacing_hz": 0.0,
                "cyclic_prefix_type": "normal",
                "total_resource_elements": 0,
                "allocated_resource_elements": 0,
                "utilization": 0.0,
                "overlap_fraction_mean": 0.0,
                "interference_penalty_db_mean": 0.0,
                "scheduled_ue_count": 0,
                "scheduled_layer_count_total": 0,
                "mu_mimo_overlap_pair_count": 0,
                "mu_mimo_group_count": 0,
                "mu_mimo_ue_count": 0,
                "adjacent_interference_pair_count": 0,
                "interference_group_count": 0,
                "interference_groups": [],
                "spatial_group_count": 0,
                "spatial_groups": [],
                "prb_symbol_occupancy_nonzero_tiles": 0,
                "prb_symbol_occupancy_max_overlap": 0,
                "ue_resource_slice_count": 0,
                "allocations": [],
            }

        allocations = []
        for allocation in list(getattr(resource_grid, "allocations", [])):
            allocations.append(
                {
                    "ue_id": str(getattr(allocation, "ue_id", "unknown_ue")),
                    "start_prb": int(getattr(allocation, "start_prb", 0)),
                    "allocated_prbs": int(getattr(allocation, "allocated_prbs", 0)),
                    "start_symbol": int(getattr(allocation, "start_symbol", 0)),
                    "allocated_symbols": int(getattr(allocation, "allocated_symbols", 0)),
                    "resource_block_indices": list(getattr(allocation, "resource_block_indices", []) or []),
                    "allocated_symbol_indices": list(getattr(allocation, "allocated_symbol_indices", []) or []),
                    "allocated_subcarrier_count": int(len(list(getattr(allocation, "allocated_subcarrier_indices", []) or []))),
                    "layer_count": int(getattr(allocation, "layer_count", 1)),
                    "resource_elements": int(getattr(allocation, "resource_elements", 0)),
                    "overlap_fraction": float(getattr(allocation, "overlap_fraction", 0.0)),
                    "interference_penalty_db": float(getattr(allocation, "interference_penalty_db", 0.0)),
                    "interference_group_id": getattr(allocation, "interference_group_id", None),
                    "spatial_group_id": getattr(allocation, "spatial_group_id", None),
                }
            )

        prb_symbol_occupancy_map = list(getattr(resource_grid, "prb_symbol_occupancy_map", []) or [])
        occupancy_values = [
            int(value)
            for row in prb_symbol_occupancy_map
            for value in list(row)
        ]

        return {
            "total_prbs": int(getattr(resource_grid, "total_prbs", 0)),
            "total_symbols": int(getattr(resource_grid, "total_symbols", 0)),
            "subcarriers_per_prb": int(getattr(resource_grid, "subcarriers_per_prb", 12)),
            "n_prbs_total": int(getattr(resource_grid, "n_prbs_total", 0)),
            "n_subcarriers_per_prb": int(getattr(resource_grid, "n_subcarriers_per_prb", 12)),
            "n_ofdm_symbols_per_slot": int(getattr(resource_grid, "n_ofdm_symbols_per_slot", 0)),
            "fft_size": int(getattr(resource_grid, "fft_size", 0)),
            "usable_subcarrier_count": int(sum(int(item) for item in getattr(resource_grid, "usable_subcarrier_mask", []))),
            "carrier_frequency_hz": float(getattr(resource_grid, "carrier_frequency_hz", 0.0)),
            "bandwidth_hz": float(getattr(resource_grid, "bandwidth_hz", 0.0)),
            "subcarrier_spacing_hz": float(getattr(resource_grid, "subcarrier_spacing_hz", 0.0)),
            "cyclic_prefix_type": getattr(resource_grid, "cyclic_prefix_type", "normal"),
            "total_resource_elements": int(getattr(resource_grid, "total_resource_elements", 0)),
            "allocated_resource_elements": int(getattr(resource_grid, "allocated_resource_elements", 0)),
            "utilization": float(getattr(resource_grid, "utilization", 0.0)),
            "overlap_fraction_mean": float(getattr(resource_grid, "overlap_fraction_mean", 0.0)),
            "interference_penalty_db_mean": float(getattr(resource_grid, "interference_penalty_db_mean", 0.0)),
            "scheduled_ue_count": int(getattr(resource_grid, "scheduled_ue_count", 0)),
            "scheduled_layer_count_total": int(getattr(resource_grid, "scheduled_layer_count_total", 0)),
            "mu_mimo_overlap_pair_count": int(getattr(resource_grid, "mu_mimo_overlap_pair_count", 0)),
            "mu_mimo_group_count": int(getattr(resource_grid, "mu_mimo_group_count", 0)),
            "mu_mimo_ue_count": int(getattr(resource_grid, "mu_mimo_ue_count", 0)),
            "adjacent_interference_pair_count": int(
                getattr(resource_grid, "adjacent_interference_pair_count", 0)
            ),
            "interference_group_count": int(len(getattr(resource_grid, "interference_groups", []))),
            "interference_groups": [
                list(group) for group in list(getattr(resource_grid, "interference_groups", []))
            ],
            "spatial_group_count": int(len(getattr(resource_grid, "spatial_groups", []))),
            "spatial_groups": [
                list(group) for group in list(getattr(resource_grid, "spatial_groups", []))
            ],
            "prb_symbol_occupancy_nonzero_tiles": int(sum(1 for value in occupancy_values if value > 0)),
            "prb_symbol_occupancy_max_overlap": int(max(occupancy_values) if occupancy_values else 0),
            "ue_resource_slice_count": int(len(getattr(resource_grid, "ue_resource_slices", {}))),
            "allocations": allocations,
        }

    def _build_ue_observable_snapshot(
        self,
        ue: Any,
        measurement_entry: Any | None,
    ) -> Dict[str, Any]:
        if measurement_entry is not None:
            return {
                "queue_bytes": int(getattr(measurement_entry, "queue_bytes", 0)),
                "throughput_mbps": float(getattr(measurement_entry, "throughput_mbps", 0.0)),
                "sinr_db": float(getattr(measurement_entry, "sinr_db", 0.0)),
                "bler": float(getattr(measurement_entry, "bler", 0.0)),
                "is_scheduled": bool(getattr(measurement_entry, "is_scheduled", False)),
                "ack": bool(getattr(measurement_entry, "ack", False)),
                "tb_size_bytes": int(getattr(measurement_entry, "tb_size_bytes", 0)),
            }

        return {
            "queue_bytes": int(self._safe_get(ue, "queue_bytes", 0)),
            "throughput_mbps": float(self._safe_get(ue, "throughput_mbps", 0.0)),
            "sinr_db": float(self._safe_get(ue, "sinr_db", 0.0)),
            "bler": float(self._safe_get(ue, "bler", 0.0)),
            "is_scheduled": bool(self._safe_get(ue, "is_scheduled", False)),
            "ack": bool(self._safe_get(ue, "ack", False)),
            "tb_size_bytes": int(self._safe_get(ue, "transport_block_size_bytes", 0)),
        }

    def _build_ue_measurement_trace(self, measurement_entry: Any | None) -> Dict[str, Any]:
        if measurement_entry is None:
            return {
                "produced_at_slot": None,
                "delivered_at_slot": None,
                "age_slots": None,
                "queue_age_slots": None,
                "queue_delay_slots_applied": None,
                "queue_produced_at_slot": None,
                "sinr_window_slots": None,
                "sinr_effective_sample_count": None,
                "sinr_noise_std_db_applied": None,
                "bler_window_tbs": None,
                "bler_sample_count": None,
                "throughput_window_slots": None,
                "throughput_granularity_slots": None,
                "last_measurement_update_slot": None,
            }

        return {
            "produced_at_slot": getattr(measurement_entry, "produced_at_slot", None),
            "delivered_at_slot": getattr(measurement_entry, "delivered_at_slot", None),
            "age_slots": getattr(measurement_entry, "age_slots", None),
            "queue_age_slots": getattr(measurement_entry, "queue_age_slots", None),
            "queue_delay_slots_applied": getattr(measurement_entry, "queue_delay_slots_applied", None),
            "queue_produced_at_slot": getattr(measurement_entry, "queue_produced_at_slot", None),
            "sinr_window_slots": getattr(measurement_entry, "sinr_window_slots", None),
            "sinr_effective_sample_count": getattr(measurement_entry, "sinr_effective_sample_count", None),
            "sinr_noise_std_db_applied": getattr(measurement_entry, "sinr_noise_std_db_applied", None),
            "bler_window_tbs": getattr(measurement_entry, "bler_window_tbs", None),
            "bler_sample_count": getattr(measurement_entry, "bler_sample_count", None),
            "throughput_window_slots": getattr(measurement_entry, "throughput_window_slots", None),
            "throughput_granularity_slots": getattr(measurement_entry, "throughput_granularity_slots", None),
            "last_measurement_update_slot": getattr(measurement_entry, "last_measurement_update_slot", None),
        }

    def _build_phy_comparison_trace(
        self,
        phy_result: Any | None,
        shadow_phy_result: Any | None,
        shadow_phy_label: str | None,
    ) -> Dict[str, Any]:
        if phy_result is None or shadow_phy_result is None:
            return {
                "enabled": False,
                "shadow_phy_label": shadow_phy_label,
                "compared_ue_count": 0,
            }

        primary_results = {
            str(getattr(item, "ue_id", "")): item
            for item in list(getattr(phy_result, "ue_results", []))
            if str(getattr(item, "ue_id", "")) != "" and bool(getattr(item, "is_scheduled", False))
        }
        shadow_results = {
            str(getattr(item, "ue_id", "")): item
            for item in list(getattr(shadow_phy_result, "ue_results", []))
            if str(getattr(item, "ue_id", "")) != "" and bool(getattr(item, "is_scheduled", False))
        }
        compared_ue_ids = sorted(set(primary_results.keys()) & set(shadow_results.keys()))
        if len(compared_ue_ids) == 0:
            return {
                "enabled": True,
                "shadow_phy_label": shadow_phy_label,
                "primary_scheduled_ue_count": int(len(primary_results)),
                "shadow_scheduled_ue_count": int(len(shadow_results)),
                "compared_ue_count": 0,
            }

        ack_matches = 0
        spatial_mode_matches = 0
        abs_sinr_gaps: list[float] = []
        abs_post_eq_sinr_gaps: list[float] = []
        abs_bler_gaps: list[float] = []
        abs_throughput_gaps: list[float] = []
        abs_tb_size_gaps: list[float] = []
        abs_spatial_gain_gaps: list[float] = []
        per_ue_diffs: list[dict[str, Any]] = []
        spatial_group_accumulator: dict[str, dict[str, Any]] = {}

        for ue_id in compared_ue_ids:
            primary_item = primary_results[ue_id]
            shadow_item = shadow_results[ue_id]
            sinr_gap_db = abs(
                float(getattr(primary_item, "sinr_db", 0.0))
                - float(getattr(shadow_item, "sinr_db", 0.0))
            )
            post_eq_sinr_gap_db = abs(
                float(getattr(primary_item, "post_eq_sinr_db", getattr(primary_item, "sinr_db", 0.0)))
                - float(getattr(shadow_item, "post_eq_sinr_db", getattr(shadow_item, "sinr_db", 0.0)))
            )
            bler_gap = abs(
                float(getattr(primary_item, "bler", 0.0)) - float(getattr(shadow_item, "bler", 0.0))
            )
            throughput_gap_mbps = abs(
                float(getattr(primary_item, "throughput_mbps", 0.0))
                - float(getattr(shadow_item, "throughput_mbps", 0.0))
            )
            tb_size_gap_bytes = abs(
                int(getattr(primary_item, "transport_block_size_bytes", 0))
                - int(getattr(shadow_item, "transport_block_size_bytes", 0))
            )
            spatial_gain_gap_db = abs(
                float(getattr(primary_item, "spatial_multiplexing_gain_db", 0.0))
                - float(getattr(shadow_item, "spatial_multiplexing_gain_db", 0.0))
            )
            ack_match = bool(getattr(primary_item, "ack", False)) == bool(getattr(shadow_item, "ack", False))
            spatial_mode_match = str(getattr(primary_item, "spatial_mode", "unknown")) == str(
                getattr(shadow_item, "spatial_mode", "unknown")
            )
            ack_matches += int(
                ack_match
            )
            spatial_mode_matches += int(
                spatial_mode_match
            )
            abs_sinr_gaps.append(sinr_gap_db)
            abs_post_eq_sinr_gaps.append(post_eq_sinr_gap_db)
            abs_bler_gaps.append(bler_gap)
            abs_throughput_gaps.append(throughput_gap_mbps)
            abs_tb_size_gaps.append(tb_size_gap_bytes)
            abs_spatial_gain_gaps.append(spatial_gain_gap_db)

            primary_spatial_group_id = getattr(primary_item, "spatial_group_id", None)
            shadow_spatial_group_id = getattr(shadow_item, "spatial_group_id", None)
            per_ue_diffs.append(
                {
                    "ue_id": ue_id,
                    "primary_spatial_group_id": primary_spatial_group_id,
                    "shadow_spatial_group_id": shadow_spatial_group_id,
                    "sinr_gap_db": sinr_gap_db,
                    "post_eq_sinr_gap_db": post_eq_sinr_gap_db,
                    "bler_gap": bler_gap,
                    "throughput_gap_mbps": throughput_gap_mbps,
                    "tb_size_gap_bytes": tb_size_gap_bytes,
                    "spatial_multiplexing_gain_gap_db": spatial_gain_gap_db,
                    "ack_match": ack_match,
                    "spatial_mode_match": spatial_mode_match,
                }
            )

            if primary_spatial_group_id is not None:
                group_key = str(primary_spatial_group_id)
                group_bucket = spatial_group_accumulator.setdefault(
                    group_key,
                    {
                        "spatial_group_id": primary_spatial_group_id,
                        "ue_ids": [],
                        "post_eq_sinr_gap_db": [],
                        "bler_gap": [],
                        "throughput_gap_mbps": [],
                        "tb_size_gap_bytes": [],
                        "ack_match_count": 0,
                        "spatial_mode_match_count": 0,
                    },
                )
                group_bucket["ue_ids"].append(ue_id)
                group_bucket["post_eq_sinr_gap_db"].append(post_eq_sinr_gap_db)
                group_bucket["bler_gap"].append(bler_gap)
                group_bucket["throughput_gap_mbps"].append(throughput_gap_mbps)
                group_bucket["tb_size_gap_bytes"].append(tb_size_gap_bytes)
                group_bucket["ack_match_count"] += int(ack_match)
                group_bucket["spatial_mode_match_count"] += int(spatial_mode_match)

        compared_count = len(compared_ue_ids)
        worst_ue_diff = max(per_ue_diffs, key=lambda item: float(item["post_eq_sinr_gap_db"]))
        spatial_group_summaries = []
        for group_key in sorted(spatial_group_accumulator, key=lambda item: int(item)):
            group_bucket = spatial_group_accumulator[group_key]
            member_count = max(1, len(group_bucket["ue_ids"]))
            spatial_group_summaries.append(
                {
                    "spatial_group_id": group_bucket["spatial_group_id"],
                    "ue_ids": list(group_bucket["ue_ids"]),
                    "member_count": member_count,
                    "mean_post_eq_sinr_gap_db": self._mean_or_none(group_bucket["post_eq_sinr_gap_db"]),
                    "mean_bler_gap": self._mean_or_none(group_bucket["bler_gap"]),
                    "mean_throughput_gap_mbps": self._mean_or_none(group_bucket["throughput_gap_mbps"]),
                    "mean_tb_size_gap_bytes": self._mean_or_none(group_bucket["tb_size_gap_bytes"]),
                    "ack_match_rate": float(group_bucket["ack_match_count"]) / float(member_count),
                    "spatial_mode_match_rate": float(group_bucket["spatial_mode_match_count"]) / float(member_count),
                }
            )

        return {
            "enabled": True,
            "shadow_phy_label": shadow_phy_label,
            "primary_scheduled_ue_count": int(len(primary_results)),
            "shadow_scheduled_ue_count": int(len(shadow_results)),
            "compared_ue_count": int(compared_count),
            "ack_match_rate": float(ack_matches) / float(compared_count),
            "spatial_mode_match_rate": float(spatial_mode_matches) / float(compared_count),
            "mean_abs_sinr_gap_db": self._mean_or_none(abs_sinr_gaps),
            "mean_abs_post_eq_sinr_gap_db": self._mean_or_none(abs_post_eq_sinr_gaps),
            "max_abs_post_eq_sinr_gap_db": float(worst_ue_diff["post_eq_sinr_gap_db"]),
            "worst_ue_id_by_post_eq_sinr_gap": worst_ue_diff["ue_id"],
            "mean_abs_bler_gap": self._mean_or_none(abs_bler_gaps),
            "mean_abs_throughput_gap_mbps": self._mean_or_none(abs_throughput_gaps),
            "mean_abs_tb_size_gap_bytes": self._mean_or_none(abs_tb_size_gaps),
            "mean_abs_spatial_multiplexing_gain_gap_db": self._mean_or_none(abs_spatial_gain_gaps),
            "per_ue_diffs": per_ue_diffs,
            "spatial_group_summaries": spatial_group_summaries,
        }

    def apply_phy_result(self, state, phy_result) -> None:
        """
        Applies one slot PHY result to simulator state.
        INPUT
        state: Simulator state object.
        phy_result: PHY result object.
        OUTPUT
        None.
        """
        scheduler_state = getattr(state, "scheduler", None)
        scheduled_ue_ids = list(getattr(scheduler_state, "scheduled_ue_ids", []))
        scheduled_ue_set = set(str(item) for item in scheduled_ue_ids)
        grants = list(getattr(scheduler_state, "grants", [])) if scheduler_state is not None else []
        grant_map = {
            str(getattr(grant, "ue_id", "")): grant
            for grant in grants
            if str(getattr(grant, "ue_id", "")) != ""
        }

        # Ensure per-UE scheduling flags are correct even if PHY returns only scheduled UE results.
        for ue in self._iter_ues(state):
            ue_id = self._safe_get(ue, "ue_id", None)
            if ue_id is None:
                continue

            ue_id_str = str(ue_id)
            self._reset_logical_channel_slot_runtime(
                ue,
                slot_id=int(self._safe_get(state, "slot_id", -1)),
            )
            connection = self._get_connection(state, ue_id_str)
            grant = grant_map.get(ue_id_str)
            is_scheduled = ue_id_str in scheduled_ue_set
            self._safe_set(ue, "is_scheduled", is_scheduled)
            self._safe_set(ue, "allocated_prbs", int(getattr(grant, "allocated_prbs", 0)) if grant is not None else 0)
            self._safe_set(ue, "allocated_symbols", int(getattr(grant, "allocated_symbols", 0)) if grant is not None else 0)
            self._safe_set(ue, "start_prb", int(getattr(grant, "start_prb", 0)) if grant is not None else 0)
            self._safe_set(ue, "start_symbol", int(getattr(grant, "start_symbol", 0)) if grant is not None else 0)
            self._safe_set(ue, "resource_block_indices", list(getattr(grant, "resource_block_indices", []) or []) if grant is not None else [])
            self._safe_set(ue, "allocated_symbol_indices", list(getattr(grant, "ofdm_symbol_indices", []) or []) if grant is not None else [])
            self._safe_set(ue, "is_retx", bool(getattr(grant, "is_retx", False)) if grant is not None else False)
            self._safe_set(ue, "priority_score", float(getattr(grant, "priority_score", 0.0)) if grant is not None else 0.0)
            self._safe_set(ue, "grant_bytes_budget", int(getattr(grant, "grant_bytes_budget", 0)) if grant is not None else 0)
            self._safe_set(ue, "grant_target_mcs", getattr(grant, "target_mcs", None) if grant is not None else None)
            self._safe_set(ue, "grant_mcs_table_id", str(getattr(grant, "mcs_table_id", "qam64")) if grant is not None else "qam64")
            self._safe_set(ue, "grant_tbs_selection_method", str(getattr(grant, "tbs_selection_method", "resource_elements")) if grant is not None else "resource_elements")
            self._safe_set(ue, "grant_dmrs_pattern_id", getattr(grant, "dmrs_pattern_id", None) if grant is not None else None)
            self._safe_set(ue, "grant_cqi_validity_age_slots", int(getattr(grant, "cqi_validity_age_slots", 0)) if grant is not None else 0)
            self._safe_set(ue, "grant_ri_validity_age_slots", int(getattr(grant, "ri_validity_age_slots", 0)) if grant is not None else 0)
            self._safe_set(ue, "grant_pmi_validity_age_slots", int(getattr(grant, "pmi_validity_age_slots", 0)) if grant is not None else 0)
            self._safe_set(
                ue,
                "grant_channel_quality_confidence",
                (
                    float(getattr(grant, "channel_quality_confidence", 1.0))
                    if grant is not None and getattr(grant, "channel_quality_confidence", None) is not None
                    else 1.0
                ),
            )
            self._safe_set(
                ue,
                "grant_link_adaptation_reason",
                str(getattr(grant, "link_adaptation_reason", "")) if grant is not None else "",
            )
            self._safe_set(ue, "spatial_group_id", getattr(grant, "spatial_group_id", None) if grant is not None else None)
            self._safe_set(ue, "spatial_group_size", int(getattr(grant, "spatial_group_size", 1)) if grant is not None else 1)
            self._safe_set(ue, "spatial_compatibility_score", float(getattr(grant, "spatial_compatibility_score", 0.0)) if grant is not None else 0.0)
            self._safe_set(
                ue,
                "scheduled_logical_channel_ids",
                [
                    str(getattr(allocation, "logical_channel_id", ""))
                    for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                    if str(getattr(allocation, "logical_channel_id", "")) != ""
                ] if grant is not None else [],
            )
            self._safe_set(
                ue,
                "scheduled_drb_ids",
                [
                    int(getattr(allocation, "drb_id", 0))
                    for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                    if getattr(allocation, "drb_id", None) is not None
                ] if grant is not None else [],
            )
            self._safe_set(
                ue,
                "scheduled_lcg_ids",
                sorted(
                    {
                        int(getattr(allocation, "lcg_id", 0))
                        for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                        if getattr(allocation, "lcg_id", None) is not None
                    }
                ) if grant is not None else [],
            )
            if connection is not None:
                self._safe_set(
                    connection,
                    "serving_beam_id",
                    int(self._safe_get(ue, "serving_beam_id", getattr(connection, "serving_beam_id", 0))),
                )

            if not is_scheduled:
                # Do not overwrite radio / PHY quality fields for unscheduled UEs.
                self._safe_set(ue, "ack", False)
                self._safe_set(ue, "throughput_mbps", 0.0)
                self._safe_set(ue, "transport_block_size_bytes", 0)
                self._safe_set(ue, "tb_size_bytes", 0)
                self._safe_set(ue, "grant_bytes_budget", 0)
                self._safe_set(ue, "grant_target_mcs", None)
                self._safe_set(ue, "grant_mcs_table_id", "qam64")
                self._safe_set(ue, "grant_tbs_selection_method", "resource_elements")
                self._safe_set(ue, "grant_dmrs_pattern_id", None)
                self._safe_set(ue, "grant_cqi_validity_age_slots", 0)
                self._safe_set(ue, "grant_ri_validity_age_slots", 0)
                self._safe_set(ue, "grant_pmi_validity_age_slots", 0)
                self._safe_set(ue, "grant_channel_quality_confidence", 1.0)
                self._safe_set(ue, "grant_link_adaptation_reason", "")
                self._safe_set(ue, "phy_mode", "system_level_approx")
                self._safe_set(ue, "waveform_backend", None)
                self._safe_set(ue, "effective_data_re_count", 0)
                self._safe_set(ue, "dmrs_overhead_fraction", 0.0)
                self._safe_set(ue, "scheduled_bytes", 0)
                self._safe_set(ue, "delivered_bytes", 0)
                self._safe_set(ue, "new_tx_bytes", 0)
                self._safe_set(ue, "retx_tx_bytes", 0)
                self._safe_set(ue, "delivered_new_bytes", 0)
                self._safe_set(ue, "delivered_retx_bytes", 0)
                self._safe_set(ue, "resource_block_indices", [])
                self._safe_set(ue, "allocated_symbol_indices", [])
                self._safe_set(ue, "allocated_subcarrier_indices", [])
                self._safe_set(ue, "resource_grid_re_count", 0)
                self._safe_set(ue, "resource_grid_overlap_fraction", 0.0)
                self._safe_set(ue, "interference_penalty_db", 0.0)
                self._safe_set(ue, "intra_cell_interference_penalty_db", 0.0)
                self._safe_set(ue, "inter_cell_interference_dbm", -120.0)
                self._safe_set(ue, "inter_cell_interference_penalty_db", 0.0)
                self._safe_set(ue, "inter_cell_interference_overlap_fraction", 0.0)
                self._safe_set(ue, "inter_cell_frequency_coupling_fraction", 0.0)
                self._safe_set(ue, "inter_cell_interference_source_count", 0)
                self._safe_set(ue, "pre_eq_sinr_db", self._safe_get(ue, "pre_eq_sinr_db", 0.0))
                self._safe_set(ue, "post_eq_sinr_db", self._safe_get(ue, "post_eq_sinr_db", 0.0))
                self._safe_set(ue, "channel_rank_limit", self._safe_get(ue, "channel_rank_limit", 1))
                self._safe_set(ue, "precoder_match_score", self._safe_get(ue, "precoder_match_score", 1.0))
                self._safe_set(ue, "spatial_multiplexing_gain_db", 0.0)
                self._safe_set(ue, "spatial_mode", "su_siso")
                self._safe_set(ue, "scheduled_logical_channel_ids", [])
                self._safe_set(ue, "scheduled_drb_ids", [])
                self._safe_set(ue, "scheduled_lcg_ids", [])

        ue_results = getattr(phy_result, "ue_results", [])
        slot_id_value = int(self._safe_get(state, "slot_id", -1))

        for ue_result in ue_results:
            ue_id = self._safe_get(ue_result, "ue_id", None)
            if ue_id is None:
                continue

            ue = self._find_ue_by_id(state, ue_id)
            if ue is None:
                continue
            grant = grant_map.get(str(ue_id))
            logical_channel_allocations = list(
                getattr(grant, "logical_channel_allocations", []) or []
            ) if grant is not None else []

            sinr_db = float(self._safe_get(ue_result, "sinr_db", 0.0))
            pre_eq_sinr_db = float(self._safe_get(ue_result, "pre_eq_sinr_db", sinr_db))
            post_eq_sinr_db = float(self._safe_get(ue_result, "post_eq_sinr_db", sinr_db))
            bler = float(self._safe_get(ue_result, "bler", 0.0))
            mcs_index = int(self._safe_get(ue_result, "mcs_index", 0))
            allocated_prbs = int(self._safe_get(ue_result, "allocated_prbs", self._safe_get(ue, "allocated_prbs", 0)))
            allocated_symbols = int(self._safe_get(ue_result, "allocated_symbols", self._safe_get(ue, "allocated_symbols", 0)))
            start_prb = int(self._safe_get(ue_result, "start_prb", self._safe_get(ue, "start_prb", 0)))
            start_symbol = int(self._safe_get(ue_result, "start_symbol", self._safe_get(ue, "start_symbol", 0)))
            is_retx = bool(self._safe_get(ue_result, "is_retx", self._safe_get(ue, "is_retx", False)))
            requested_bytes = int(self._safe_get(ue_result, "requested_bytes", 0))
            grant_bytes_budget = int(self._safe_get(ue_result, "grant_bytes_budget", self._safe_get(ue, "grant_bytes_budget", 0)))
            grant_mcs_table_id = str(
                self._safe_get(ue_result, "mcs_table_id", self._safe_get(ue, "grant_mcs_table_id", "qam64"))
            )
            grant_tbs_selection_method = str(
                self._safe_get(
                    ue_result,
                    "tbs_selection_method",
                    self._safe_get(ue, "grant_tbs_selection_method", "resource_elements"),
                )
            )
            grant_dmrs_pattern_id = self._safe_get(
                ue_result,
                "dmrs_pattern_id",
                self._safe_get(ue, "grant_dmrs_pattern_id", None),
            )
            grant_cqi_validity_age_slots = int(
                self._safe_get(
                    ue_result,
                    "cqi_validity_age_slots",
                    self._safe_get(ue, "grant_cqi_validity_age_slots", 0),
                )
            )
            grant_ri_validity_age_slots = int(
                self._safe_get(
                    ue_result,
                    "ri_validity_age_slots",
                    self._safe_get(ue, "grant_ri_validity_age_slots", 0),
                )
            )
            grant_pmi_validity_age_slots = int(
                self._safe_get(
                    ue_result,
                    "pmi_validity_age_slots",
                    self._safe_get(ue, "grant_pmi_validity_age_slots", 0),
                )
            )
            grant_channel_quality_confidence = self._safe_get(
                ue_result,
                "channel_quality_confidence",
                self._safe_get(ue, "grant_channel_quality_confidence", 1.0),
            )
            grant_link_adaptation_reason = str(
                self._safe_get(
                    ue_result,
                    "link_adaptation_reason",
                    self._safe_get(ue, "grant_link_adaptation_reason", ""),
                )
            )
            phy_mode = str(
                self._safe_get(
                    ue_result,
                    "phy_mode",
                    self._safe_get(ue, "phy_mode", "system_level_approx"),
                )
            )
            waveform_backend = self._safe_get(
                ue_result,
                "waveform_backend",
                self._safe_get(ue, "waveform_backend", None),
            )
            effective_data_re_count = int(
                self._safe_get(
                    ue_result,
                    "effective_data_re_count",
                    self._safe_get(ue, "effective_data_re_count", 0),
                )
            )
            dmrs_overhead_fraction = float(
                self._safe_get(
                    ue_result,
                    "dmrs_overhead_fraction",
                    self._safe_get(ue, "dmrs_overhead_fraction", 0.0),
                )
            )
            resource_grid_re_count = int(self._safe_get(ue_result, "resource_grid_re_count", 0))
            resource_grid_overlap_fraction = float(self._safe_get(ue_result, "resource_grid_overlap_fraction", 0.0))
            interference_penalty_db = float(self._safe_get(ue_result, "interference_penalty_db", 0.0))
            intra_cell_interference_penalty_db = float(
                self._safe_get(ue_result, "intra_cell_interference_penalty_db", 0.0)
            )
            inter_cell_interference_dbm = float(
                self._safe_get(ue_result, "inter_cell_interference_dbm", -120.0)
            )
            inter_cell_interference_penalty_db = float(
                self._safe_get(ue_result, "inter_cell_interference_penalty_db", 0.0)
            )
            inter_cell_interference_overlap_fraction = float(
                self._safe_get(ue_result, "inter_cell_interference_overlap_fraction", 0.0)
            )
            inter_cell_frequency_coupling_fraction = float(
                self._safe_get(ue_result, "inter_cell_frequency_coupling_fraction", 0.0)
            )
            inter_cell_interference_source_count = int(
                self._safe_get(ue_result, "inter_cell_interference_source_count", 0)
            )

            transport_block_size_bytes = int(
                self._safe_get(
                    ue_result,
                    "transport_block_size_bytes",
                    self._safe_get(ue_result, "tb_size_bytes", 0),
                )
            )
            scheduled_bytes = int(self._safe_get(ue_result, "scheduled_bytes", transport_block_size_bytes))
            delivered_bytes = int(
                self._safe_get(
                    ue_result,
                    "delivered_bytes",
                    scheduled_bytes if bool(self._safe_get(ue_result, "ack", False)) else 0,
                )
            )

            ack = bool(self._safe_get(ue_result, "ack", False))
            cqi = int(self._safe_get(ue_result, "cqi", 1))
            rank = int(self._safe_get(ue_result, "rank", 1))
            pmi = int(self._safe_get(ue_result, "pmi", 0))
            ri = int(self._safe_get(ue_result, "ri", 1))
            channel_rank_limit = int(self._safe_get(ue_result, "channel_rank_limit", rank))
            precoder_match_score = float(self._safe_get(ue_result, "precoder_match_score", 1.0))
            spatial_group_id = self._safe_get(ue_result, "spatial_group_id", self._safe_get(ue, "spatial_group_id", None))
            spatial_group_size = int(self._safe_get(ue_result, "spatial_group_size", self._safe_get(ue, "spatial_group_size", 1)))
            spatial_compatibility_score = float(
                self._safe_get(ue_result, "spatial_compatibility_score", self._safe_get(ue, "spatial_compatibility_score", 0.0))
            )
            spatial_multiplexing_gain_db = float(
                self._safe_get(ue_result, "spatial_multiplexing_gain_db", self._safe_get(ue, "spatial_multiplexing_gain_db", 0.0))
            )
            spatial_mode = str(self._safe_get(ue_result, "spatial_mode", self._safe_get(ue, "spatial_mode", "su_siso")))
            is_scheduled = bool(self._safe_get(ue_result, "is_scheduled", False))

            throughput_mbps = float(self._safe_get(ue_result, "throughput_mbps", 0.0))

            if not is_scheduled:
                continue

            self._apply_logical_channel_scheduled_bytes(
                ue,
                logical_channel_allocations,
                slot_id=slot_id_value,
            )

            if throughput_mbps <= 0.0 and transport_block_size_bytes > 0 and ack:
                slot_duration_ms = float(self._safe_get(state, "slot_duration_ms", 1.0))
                slot_duration_ms = max(slot_duration_ms, 1e-6)
                throughput_mbps = (transport_block_size_bytes * 8.0) / 1e6 / (slot_duration_ms / 1000.0)

            self._safe_set(ue, "sinr_db", sinr_db)
            self._safe_set(ue, "pre_eq_sinr_db", pre_eq_sinr_db)
            self._safe_set(ue, "post_eq_sinr_db", post_eq_sinr_db)
            self._safe_set(ue, "bler", bler)
            self._safe_set(ue, "mcs_index", mcs_index)
            self._safe_set(ue, "allocated_prbs", allocated_prbs)
            self._safe_set(ue, "allocated_symbols", allocated_symbols)
            self._safe_set(ue, "start_prb", start_prb)
            self._safe_set(ue, "start_symbol", start_symbol)
            resource_block_indices = list(self._safe_get(ue, "resource_block_indices", []) or [])
            allocated_symbol_indices = list(self._safe_get(ue, "allocated_symbol_indices", []) or [])
            if not resource_block_indices:
                resource_block_indices = list(range(start_prb, start_prb + max(0, allocated_prbs)))
                self._safe_set(ue, "resource_block_indices", resource_block_indices)
            if not allocated_symbol_indices:
                allocated_symbol_indices = list(range(start_symbol, start_symbol + max(0, allocated_symbols)))
                self._safe_set(ue, "allocated_symbol_indices", allocated_symbol_indices)
            allocated_subcarrier_indices = []
            for prb_index in resource_block_indices:
                base_subcarrier = int(prb_index) * 12
                allocated_subcarrier_indices.extend(range(base_subcarrier, base_subcarrier + 12))
            self._safe_set(ue, "allocated_subcarrier_indices", allocated_subcarrier_indices)
            self._safe_set(ue, "is_retx", is_retx)
            self._safe_set(ue, "transport_block_size_bytes", transport_block_size_bytes)
            self._safe_set(ue, "tb_size_bytes", transport_block_size_bytes)
            self._safe_set(ue, "grant_bytes_budget", grant_bytes_budget)
            self._safe_set(ue, "grant_mcs_table_id", grant_mcs_table_id)
            self._safe_set(ue, "grant_tbs_selection_method", grant_tbs_selection_method)
            self._safe_set(ue, "grant_dmrs_pattern_id", grant_dmrs_pattern_id)
            self._safe_set(ue, "grant_cqi_validity_age_slots", grant_cqi_validity_age_slots)
            self._safe_set(ue, "grant_ri_validity_age_slots", grant_ri_validity_age_slots)
            self._safe_set(ue, "grant_pmi_validity_age_slots", grant_pmi_validity_age_slots)
            self._safe_set(
                ue,
                "grant_channel_quality_confidence",
                float(grant_channel_quality_confidence) if grant_channel_quality_confidence is not None else 1.0,
            )
            self._safe_set(ue, "grant_link_adaptation_reason", grant_link_adaptation_reason)
            self._safe_set(ue, "phy_mode", phy_mode)
            self._safe_set(ue, "waveform_backend", waveform_backend)
            self._safe_set(ue, "effective_data_re_count", effective_data_re_count)
            self._safe_set(ue, "dmrs_overhead_fraction", dmrs_overhead_fraction)
            self._safe_set(ue, "scheduled_bytes", scheduled_bytes)
            self._safe_set(ue, "delivered_bytes", delivered_bytes)
            self._safe_set(ue, "new_tx_bytes", scheduled_bytes if not is_retx else 0)
            self._safe_set(ue, "retx_tx_bytes", scheduled_bytes if is_retx else 0)
            self._safe_set(ue, "delivered_new_bytes", delivered_bytes if (ack and not is_retx) else 0)
            self._safe_set(ue, "delivered_retx_bytes", delivered_bytes if (ack and is_retx) else 0)
            self._safe_set(ue, "ack", ack)
            self._safe_set(ue, "cqi", cqi)
            self._safe_set(ue, "rank", rank)
            self._safe_set(ue, "pmi", pmi)
            self._safe_set(ue, "ri", ri)
            self._safe_set(ue, "channel_rank_limit", channel_rank_limit)
            self._safe_set(ue, "precoder_match_score", precoder_match_score)
            self._safe_set(ue, "spatial_group_id", spatial_group_id)
            self._safe_set(ue, "spatial_group_size", spatial_group_size)
            self._safe_set(ue, "spatial_compatibility_score", spatial_compatibility_score)
            self._safe_set(ue, "spatial_multiplexing_gain_db", spatial_multiplexing_gain_db)
            self._safe_set(ue, "spatial_mode", spatial_mode)
            self._safe_set(ue, "is_scheduled", is_scheduled)
            self._safe_set(ue, "throughput_mbps", throughput_mbps if (is_scheduled and ack) else 0.0)
            self._safe_set(ue, "resource_grid_re_count", resource_grid_re_count)
            self._safe_set(ue, "resource_grid_overlap_fraction", resource_grid_overlap_fraction)
            self._safe_set(ue, "interference_penalty_db", interference_penalty_db)
            self._safe_set(ue, "intra_cell_interference_penalty_db", intra_cell_interference_penalty_db)
            self._safe_set(ue, "inter_cell_interference_dbm", inter_cell_interference_dbm)
            self._safe_set(ue, "inter_cell_interference_penalty_db", inter_cell_interference_penalty_db)
            self._safe_set(
                ue,
                "inter_cell_interference_overlap_fraction",
                inter_cell_interference_overlap_fraction,
            )
            self._safe_set(
                ue,
                "inter_cell_frequency_coupling_fraction",
                inter_cell_frequency_coupling_fraction,
            )
            self._safe_set(
                ue,
                "inter_cell_interference_source_count",
                inter_cell_interference_source_count,
            )

            harq = self._safe_get(ue, "harq", None)
            if harq is not None and is_scheduled:
                try:
                    previous_pending_retx_bytes = int(getattr(harq, "pending_retx_bytes", 0))
                    previous_retx_count = int(getattr(harq, "retransmission_count", 0))
                    previous_pending_retx_logical_channels = list(
                        getattr(harq, "pending_retx_logical_channels", []) or []
                    )
                    process_id_value = self._safe_get(ue_result, "harq_process_id", None)
                    if process_id_value is None:
                        process_id_value = self._safe_get(harq, "process_id", None)
                    redundancy_version_value = self._safe_get(ue_result, "harq_redundancy_version", None)
                    if redundancy_version_value is None:
                        redundancy_version_value = self._safe_get(harq, "redundancy_version", 0)
                    redundancy_version_value = int(redundancy_version_value or 0)
                    ndi_value = self._safe_get(ue_result, "harq_ndi", None)
                    if ndi_value is None:
                        ndi_value = self._safe_get(harq, "ndi", 0)
                    ndi_value = int(ndi_value or 0)
                    retransmission_round_value = self._safe_get(
                        ue_result,
                        "harq_retransmission_round",
                        None,
                    )
                    if retransmission_round_value is None:
                        retransmission_round_value = previous_retx_count if bool(is_retx) else 0
                    retransmission_round_value = int(retransmission_round_value or 0)
                    active_processes = self._get_harq_processes(harq)
                    matching_process = None
                    if process_id_value is not None:
                        for process in active_processes:
                            if int(getattr(process, "process_id", -1)) == int(process_id_value):
                                matching_process = process
                                break
                    rv_sequence = self._get_harq_rv_sequence(matching_process)
                    max_retransmission_rounds = int(
                        getattr(
                            matching_process,
                            "max_retransmission_rounds",
                            getattr(harq, "max_retransmission_rounds", 4),
                        )
                        or 4
                    )
                    retx_deadline_slot = getattr(matching_process, "retx_deadline_slot", None)
                    timeout_slot = getattr(matching_process, "timeout_slot", None)
                    tb_id_value = getattr(matching_process, "tb_id", None)
                    service_type_value = getattr(
                        matching_process,
                        "service_type",
                        str(getattr(ue, "traffic_type", "")),
                    )
                    reserved_at_slot = getattr(matching_process, "reserved_at_slot", slot_id_value)
                    harq.last_tx_bytes = scheduled_bytes
                    harq.last_tx_was_retx = bool(is_retx)
                    harq.last_delivered_bytes = delivered_bytes
                    harq.is_waiting_ack = False
                    harq.process_id = process_id_value if process_id_value is not None else getattr(harq, "process_id", 0)
                    harq.redundancy_version = int(redundancy_version_value)
                    harq.ndi = int(ndi_value)
                    harq.last_harq_process_id = process_id_value
                    harq.last_retransmission_round = int(retransmission_round_value)

                    if ack:
                        harq.last_decision = "ACK"
                        harq.last_flush_reason = None
                        if is_retx:
                            self._apply_logical_channel_delivery(
                                ue,
                                logical_channel_allocations,
                                delivered_bytes=delivered_bytes,
                                slot_id=slot_id_value,
                            )
                            reduced_pending_retx_logical_channels = self._reduce_pending_harq_logical_channels(
                                previous_pending_retx_logical_channels,
                                delivered_bytes=delivered_bytes,
                            )
                            remaining_pending_retx_bytes = max(0, previous_pending_retx_bytes - delivered_bytes)
                            if len(reduced_pending_retx_logical_channels) > 0:
                                remaining_pending_retx_bytes = int(
                                    sum(
                                        int(getattr(entry, "pending_bytes", 0))
                                        for entry in reduced_pending_retx_logical_channels
                                    )
                                )
                            if matching_process is not None:
                                remaining_process_bytes = remaining_pending_retx_bytes
                                if remaining_process_bytes <= 0:
                                    active_processes = [
                                        process
                                        for process in active_processes
                                        if int(getattr(process, "process_id", -1)) != int(getattr(matching_process, "process_id", -1))
                                    ]
                                else:
                                    updated_processes = []
                                    for process in active_processes:
                                        if int(getattr(process, "process_id", -1)) == int(getattr(matching_process, "process_id", -1)):
                                            updated_processes.append(
                                                self._clone_harq_process(
                                                    process,
                                                    process_id=int(getattr(process, "process_id", 0)),
                                                    overrides={
                                                        "is_busy": int(remaining_process_bytes) > 0,
                                                        "pending_bytes": int(remaining_process_bytes),
                                                        "last_tx_bytes": int(scheduled_bytes),
                                                        "last_delivered_bytes": int(delivered_bytes),
                                                        "is_waiting_ack": False,
                                                        "pending_logical_channels": list(reduced_pending_retx_logical_channels),
                                                        "flush_reason": None,
                                                        "last_tx_slot": int(slot_id_value),
                                                    },
                                                )
                                            )
                                        else:
                                            updated_processes.append(process)
                                    active_processes = updated_processes
                        else:
                            self._apply_logical_channel_delivery(
                                ue,
                                logical_channel_allocations,
                                delivered_bytes=delivered_bytes,
                                slot_id=slot_id_value,
                            )
                            if process_id_value is not None:
                                active_processes = [
                                    process
                                    for process in active_processes
                                    if int(getattr(process, "process_id", -1)) != int(process_id_value)
                                ]
                        self._set_harq_processes(harq, active_processes)
                        self._synchronize_harq_legacy_fields(harq)
                    else:
                        harq.last_decision = "NACK"
                        if is_retx:
                            pending_retx_logical_channels = (
                                previous_pending_retx_logical_channels
                                if len(previous_pending_retx_logical_channels) > 0
                                else self._build_harq_pending_from_allocations(
                                    logical_channel_allocations,
                                    process_id=(
                                        int(process_id_value)
                                        if process_id_value is not None
                                        else None
                                    ),
                                    redundancy_version=int(redundancy_version_value),
                                    ndi=int(ndi_value),
                                    retransmission_round=int(retransmission_round_value),
                                )
                            )
                            pending_retx_bytes = max(
                                previous_pending_retx_bytes,
                                int(
                                    sum(
                                        int(getattr(entry, "pending_bytes", 0))
                                        for entry in pending_retx_logical_channels
                                    )
                                ),
                                scheduled_bytes,
                            )
                            next_retransmission_round = int(retransmission_round_value) + 1
                            flush_reason = None
                            if timeout_slot is not None and int(slot_id_value) >= int(timeout_slot):
                                flush_reason = "timeout"
                            elif retx_deadline_slot is not None and int(slot_id_value) >= int(retx_deadline_slot):
                                flush_reason = "retx_deadline"
                            elif int(next_retransmission_round) > int(max_retransmission_rounds):
                                flush_reason = "max_rounds"
                            if flush_reason is not None:
                                self._record_harq_flush(harq, reason=flush_reason, slot_id=int(slot_id_value))
                                harq.last_decision = f"NACK_{str(flush_reason).upper()}"
                                active_processes = [
                                    process
                                    for process in active_processes
                                    if int(getattr(process, "process_id", -1)) != int(
                                        process_id_value if process_id_value is not None else getattr(harq, "process_id", 0)
                                    )
                                ]
                            else:
                                next_redundancy_version = int(
                                    rv_sequence[
                                        min(
                                            int(next_retransmission_round),
                                            len(rv_sequence) - 1,
                                        )
                                    ]
                                )
                                updated_process = self._clone_harq_process(
                                    matching_process,
                                    process_id=int(
                                        process_id_value
                                        if process_id_value is not None
                                        else getattr(harq, "process_id", 0)
                                    ),
                                    overrides={
                                        "is_busy": True,
                                        "ndi": int(ndi_value),
                                        "tb_id": tb_id_value,
                                        "redundancy_version": int(next_redundancy_version),
                                        "retransmission_round": int(next_retransmission_round),
                                        "pending_bytes": int(pending_retx_bytes),
                                        "last_tx_bytes": int(scheduled_bytes),
                                        "last_delivered_bytes": int(delivered_bytes),
                                        "is_waiting_ack": False,
                                        "retx_deadline_slot": retx_deadline_slot,
                                        "timeout_slot": timeout_slot,
                                        "flush_reason": None,
                                        "service_type": service_type_value,
                                        "max_retransmission_rounds": int(max_retransmission_rounds),
                                        "reserved_at_slot": reserved_at_slot,
                                        "last_tx_slot": int(slot_id_value),
                                        "pending_logical_channels": list(pending_retx_logical_channels),
                                    },
                                )
                                if matching_process is not None:
                                    active_processes = [
                                        updated_process
                                        if int(getattr(process, "process_id", -1)) == int(getattr(matching_process, "process_id", -1))
                                        else process
                                        for process in active_processes
                                    ]
                                else:
                                    active_processes.append(updated_process)
                        else:
                            pending_retx_logical_channels = self._build_harq_pending_from_allocations(
                                logical_channel_allocations,
                                process_id=(
                                    int(process_id_value)
                                    if process_id_value is not None
                                    else None
                                ),
                                redundancy_version=int(self._next_harq_redundancy_version(1)),
                                ndi=int(ndi_value),
                                retransmission_round=1,
                            )
                            pending_retx_bytes = int(
                                sum(
                                    int(getattr(entry, "pending_bytes", 0))
                                    for entry in pending_retx_logical_channels
                                )
                            ) if len(pending_retx_logical_channels) > 0 else scheduled_bytes
                            flush_reason = None
                            if timeout_slot is not None and int(slot_id_value) >= int(timeout_slot):
                                flush_reason = "timeout"
                            elif retx_deadline_slot is not None and int(slot_id_value) >= int(retx_deadline_slot):
                                flush_reason = "retx_deadline"
                            elif 1 > int(max_retransmission_rounds):
                                flush_reason = "max_rounds"
                            if flush_reason is not None:
                                self._record_harq_flush(harq, reason=flush_reason, slot_id=int(slot_id_value))
                                harq.last_decision = f"NACK_{str(flush_reason).upper()}"
                                active_processes = [
                                    process
                                    for process in active_processes
                                    if int(getattr(process, "process_id", -1)) != int(
                                        process_id_value if process_id_value is not None else getattr(harq, "process_id", 0)
                                    )
                                ]
                            else:
                                updated_process = self._clone_harq_process(
                                    matching_process,
                                    process_id=int(
                                        process_id_value
                                        if process_id_value is not None
                                        else getattr(harq, "process_id", 0)
                                    ),
                                    overrides={
                                        "is_busy": True,
                                        "ndi": int(ndi_value),
                                        "tb_id": tb_id_value,
                                        "redundancy_version": int(self._next_harq_redundancy_version(1)),
                                        "retransmission_round": 1,
                                        "pending_bytes": int(pending_retx_bytes),
                                        "last_tx_bytes": int(scheduled_bytes),
                                        "last_delivered_bytes": int(delivered_bytes),
                                        "is_waiting_ack": False,
                                        "retx_deadline_slot": retx_deadline_slot,
                                        "timeout_slot": timeout_slot,
                                        "flush_reason": None,
                                        "service_type": service_type_value,
                                        "max_retransmission_rounds": int(max_retransmission_rounds),
                                        "reserved_at_slot": reserved_at_slot,
                                        "last_tx_slot": int(slot_id_value),
                                        "pending_logical_channels": list(pending_retx_logical_channels),
                                    },
                                )
                                active_processes = [
                                    process
                                    for process in active_processes
                                    if int(getattr(process, "process_id", -1)) != int(getattr(updated_process, "process_id", -1))
                                ]
                                active_processes.append(updated_process)
                        self._set_harq_processes(harq, active_processes)
                        self._synchronize_harq_legacy_fields(harq)
                except Exception:
                    pass

            connection = self._get_connection(state, str(ue_id))
            link = self._get_link(state, str(ue_id))
            beamforming = self._get_beamforming(state, str(ue_id))
            if connection is not None and is_scheduled:
                slot_id_value = int(self._safe_get(state, "slot_id", -1))
                self._safe_set(connection, "last_scheduled_slot", slot_id_value)
                self._safe_set(
                    connection,
                    "serving_beam_id",
                    int(self._safe_get(ue, "serving_beam_id", getattr(connection, "serving_beam_id", 0))),
                )
                self._safe_set(connection, "rank_indicator", max(1, int(rank)))
                self._safe_set(connection, "pmi", int(pmi))
                self._safe_set(connection, "cqi", int(cqi))
                if ack:
                    self._safe_set(connection, "last_ack_slot", slot_id_value)
                    self._safe_set(connection, "consecutive_nack_count", 0)
                    self._safe_set(connection, "out_of_sync_reason", None)
                    if str(self._safe_get(connection, "connection_state", "connected")) != "handover_pending":
                        self._safe_set(connection, "connection_state", "connected")
                else:
                    next_nack_count = int(self._safe_get(connection, "consecutive_nack_count", 0)) + 1
                    self._safe_set(connection, "consecutive_nack_count", next_nack_count)
                    if next_nack_count >= 8:
                        self._safe_set(connection, "connection_state", "out_of_sync")
                        self._safe_set(connection, "out_of_sync_reason", "consecutive_nack_threshold")

                active_harq_processes: list[int] = []
                if harq is not None:
                    active_harq_processes = [
                        int(getattr(process, "process_id", 0))
                        for process in self._get_harq_processes(harq)
                        if int(getattr(process, "pending_bytes", 0)) > 0
                    ]
                    if len(active_harq_processes) == 0:
                        process_id = getattr(harq, "process_id", None)
                        pending_retx_bytes = int(getattr(harq, "pending_retx_bytes", 0))
                        if process_id is not None and pending_retx_bytes > 0:
                            active_harq_processes.append(int(process_id))
                self._safe_set(connection, "active_harq_processes", active_harq_processes)
            if link is not None and is_scheduled:
                self._safe_set(link, "last_pre_eq_sinr_db", float(pre_eq_sinr_db))
                self._safe_set(link, "last_post_eq_sinr_db", float(sinr_db))
                self._safe_set(link, "cqi", int(cqi))
                self._safe_set(link, "ri", int(ri))
                self._safe_set(link, "pmi", int(pmi))
            if beamforming is not None and is_scheduled:
                self._safe_set(beamforming, "serving_beam_id", int(self._safe_get(ue, "serving_beam_id", 0)))
                self._safe_set(beamforming, "rank", max(1, int(rank)))
                self._safe_set(beamforming, "layer_count", max(1, int(self._safe_get(ue_result, "num_layers", getattr(beamforming, "layer_count", 1)))))
                self._safe_set(beamforming, "combiner_id", int(pmi))
                self._safe_set(beamforming, "beam_gain_db", float(self._safe_get(ue_result, "beam_gain_db", 0.0)))
                self._safe_set(beamforming, "beam_direction_deg", float(int(self._safe_get(ue, "serving_beam_id", 0)) * 30.0))
                self._safe_set(beamforming, "precoder_match_score", float(precoder_match_score))
                self._safe_set(beamforming, "post_eq_sinr_db", float(post_eq_sinr_db))
                self._safe_set(beamforming, "spatial_group_id", spatial_group_id)
                self._safe_set(
                    beamforming,
                    "precoding_matrix_label",
                    f"cb{int(getattr(beamforming, 'precoder_id', 0))}-r{max(1, int(rank))}-b{int(self._safe_get(ue, 'serving_beam_id', 0))}",
                )

            logical_channels = self._get_logical_channels(ue)
            if len(logical_channels) > 0:
                self._recompute_ue_queue_bytes(
                    ue,
                    slot_id=slot_id_value,
                )
            elif is_scheduled and delivered_bytes > 0:
                current_queue_bytes = int(self._safe_get(ue, "queue_bytes", 0))
                new_queue_bytes = max(0, current_queue_bytes - delivered_bytes)
                self._safe_set(ue, "queue_bytes", new_queue_bytes)

            debug_print(
                f"StateUpdater applied UE result. "
                f"ue_id={ue_id}, sinr_db={sinr_db:.2f}, bler={bler:.4f}, "
                f"requested_bytes={requested_bytes}, grant_bytes_budget={grant_bytes_budget}, scheduled_bytes={scheduled_bytes}, delivered_bytes={delivered_bytes}, "
                f"grant={allocated_prbs}prb/{allocated_symbols}sym, "
                f"{'retx' if is_retx else 'new'} ack={ack}."
            )

        if len(scheduled_ue_set) == 0:
            self._safe_set(state, "avg_sinr_db", None)
            self._safe_set(state, "avg_bler", None)
        else:
            self._safe_set(state, "avg_sinr_db", float(self._safe_get(phy_result, "avg_sinr_db", 0.0)))
            self._safe_set(state, "avg_bler", float(self._safe_get(phy_result, "avg_bler", 0.0)))
        self._safe_set(
            state,
            "total_tb_size_bytes",
            int(self._safe_get(phy_result, "total_tb_size_bytes", 0)),
        )
        self._safe_set(state, "last_phy_result", phy_result)
        self._update_resource_grid_state(state, phy_result)

    def record_snapshot(self, state, observation: Any | None = None) -> None:
        """
        Records one state snapshot after the current slot.
        INPUT
        state: Simulator state object.
        OUTPUT
        None.
        """
        if not hasattr(state, "snapshots") or getattr(state, "snapshots") is None:
            state.snapshots = []

        if observation is None:
            observation = build_observation(state)

        temp_slot_records: List[Dict[str, Any]] = []
        self.record_slot_metrics(
            state=state,
            observation=observation,
            slot_records=temp_slot_records,
            shadow_phy_result=getattr(state, "last_shadow_phy_result", None),
            shadow_phy_label=getattr(state, "last_shadow_phy_label", None),
        )
        metric_item = temp_slot_records[0] if len(temp_slot_records) > 0 else {}

        measurement_state = getattr(state, "measurement", None)
        measurement_map = (
            dict(getattr(measurement_state, "ue_measurements", {}))
            if measurement_state is not None else {}
        )
        ues_snapshot: list[dict[str, Any]] = []

        for ue in self._iter_ues(state):
            ue_id = str(self._safe_get(ue, "ue_id", "unknown_ue"))
            measurement_entry = measurement_map.get(ue_id)
            connection = self._get_connection(state, ue_id)
            link = self._get_link(state, ue_id)
            beamforming = self._get_beamforming(state, ue_id)
            ues_snapshot.append(
                {
                    "ue_id": ue_id,
                    "slice_id": str(self._safe_get(ue, "slice_id", "")),
                    "traffic_type": str(self._safe_get(ue, "traffic_type", "")),
                    "latency_budget_ms": float(self._safe_get(ue, "latency_budget_ms", 0.0)),
                    "priority": int(self._safe_get(ue, "priority", 0)),
                    "true_state": self._build_ue_true_snapshot(
                        ue,
                        slot_id=int(self._safe_get(state, "slot_id", -1)),
                    ),
                    "observable_state": self._build_ue_observable_snapshot(ue, measurement_entry),
                    "measurement_trace": self._build_ue_measurement_trace(measurement_entry),
                    "connection": self._build_ue_connection_snapshot(connection, ue),
                    "link": self._build_ue_link_snapshot(link, state),
                    "beamforming": self._build_ue_beamforming_snapshot(beamforming, ue, state),
                    "logical_channels": self._build_ue_logical_channel_snapshot(ue),
                    "logical_channel_groups": self._build_ue_logical_channel_group_snapshot(
                        ue,
                        slot_id=int(self._safe_get(state, "slot_id", -1)),
                    ),
                }
            )

        snapshot: Dict[str, Any] = {
            "slot_id": int(self._safe_get(state, "slot_id", -1)),
            "sim_time_ms": float(self._safe_get(state, "sim_time_ms", 0.0)),
            "antenna": self._build_antenna_snapshot(state),
            "resource_grid": self._build_resource_grid_snapshot(state),
            "true_metrics": deepcopy(dict(metric_item.get("true_metrics", {}))),
            "observable_metrics": deepcopy(dict(metric_item.get("observable_metrics", {}))),
            "measurement_trace": deepcopy(dict(metric_item.get("measurement_trace", {}))),
            "control_trace": deepcopy(dict(metric_item.get("control_trace", {}))),
            "scheduler_trace": deepcopy(dict(metric_item.get("scheduler_trace", {}))),
            "phy_comparison_trace": deepcopy(dict(metric_item.get("phy_comparison_trace", {}))),
            "ues": ues_snapshot,
        }

        state.snapshots.append(deepcopy(snapshot))

    def record_slot_metrics(
        self,
        state,
        observation: Any | None = None,
        slot_records: List[Dict[str, Any]] | None = None,
        *,
        slot_id: int | None = None,
        phy_result: Any | None = None,
        shadow_phy_result: Any | None = None,
        shadow_phy_label: str | None = None,
    ) -> None:
        """
        Records one slot-level metric entry.
        INPUT
        state: Simulator state object.
        slot_id: Current slot id.
        phy_result: PHY result object.
        OUTPUT
        None.
        """
        if slot_records is None:
            if not hasattr(state, "slot_metrics") or getattr(state, "slot_metrics") is None:
                state.slot_metrics = []
            slot_records = state.slot_metrics

        slot_id_value = int(slot_id if slot_id is not None else self._safe_get(state, "slot_id", -1))
        sim_time_ms = float(self._safe_get(state, "sim_time_ms", 0.0))

        if phy_result is None:
            phy_result = self._safe_get(state, "last_phy_result", None)
        if shadow_phy_result is None:
            shadow_phy_result = self._safe_get(state, "last_shadow_phy_result", None)
        if shadow_phy_label is None:
            shadow_phy_label = self._safe_get(state, "last_shadow_phy_label", None)

        ue_list = self._iter_ues(state)
        total_queue_bytes = sum(int(self._safe_get(ue, "queue_bytes", 0)) for ue in ue_list)
        total_throughput_mbps = sum(float(self._safe_get(ue, "throughput_mbps", 0.0)) for ue in ue_list)
        total_pending_retx_bytes = sum(int(getattr(getattr(ue, "harq", None), "pending_retx_bytes", 0)) for ue in ue_list)
        harq_flush_count_total = int(
            sum(int(getattr(getattr(ue, "harq", None), "harq_flush_count", 0)) for ue in ue_list)
        )
        active_harq_process_count_total = int(
            sum(
                len(self._get_harq_processes(getattr(ue, "harq", None)))
                for ue in ue_list
                if getattr(ue, "harq", None) is not None
            )
        )
        harq_retransmission_rounds = [
            int(getattr(getattr(ue, "harq", None), "retransmission_count", 0))
            for ue in ue_list
            if getattr(ue, "harq", None) is not None
        ]
        phy_modes_scheduled = sorted(
            {
                str(self._safe_get(ue, "phy_mode", "system_level_approx"))
                for ue in ue_list
                if bool(self._safe_get(ue, "is_scheduled", False))
            }
        )
        waveform_backends_scheduled = sorted(
            {
                str(self._safe_get(ue, "waveform_backend", ""))
                for ue in ue_list
                if bool(self._safe_get(ue, "is_scheduled", False))
                and self._safe_get(ue, "waveform_backend", None) is not None
                and str(self._safe_get(ue, "waveform_backend", "")) != ""
            }
        )
        avg_effective_data_re_count_scheduled = self._mean_or_none(
            [
                float(self._safe_get(ue, "effective_data_re_count", 0))
                for ue in ue_list
                if bool(self._safe_get(ue, "is_scheduled", False))
            ]
        )
        avg_dmrs_overhead_fraction_scheduled = self._mean_or_none(
            [
                float(self._safe_get(ue, "dmrs_overhead_fraction", 0.0))
                for ue in ue_list
                if bool(self._safe_get(ue, "is_scheduled", False))
            ]
        )
        total_logical_channel_count = int(sum(len(self._get_logical_channels(ue)) for ue in ue_list))
        total_logical_channel_group_count = int(
            sum(
                len(
                    self._recompute_logical_channel_groups(
                        ue,
                        slot_id=int(self._safe_get(state, "slot_id", -1)),
                    )
                )
                for ue in ue_list
            )
        )
        all_logical_channel_groups = [
            logical_channel_group
            for ue in ue_list
            for logical_channel_group in self._recompute_logical_channel_groups(
                ue,
                slot_id=int(self._safe_get(state, "slot_id", -1)),
            )
        ]
        connection_map = getattr(state, "connections", {})
        if isinstance(connection_map, dict) and len(connection_map) > 0:
            connected_ue_count = int(
                sum(
                    1
                    for connection in connection_map.values()
                    if str(getattr(connection, "connection_state", "connected")) == "connected"
                )
            )
            out_of_sync_ue_count = int(
                sum(
                    1
                    for connection in connection_map.values()
                    if str(getattr(connection, "connection_state", "connected")) == "out_of_sync"
                )
            )
            handover_pending_ue_count = int(
                sum(
                    1
                    for connection in connection_map.values()
                    if str(getattr(connection, "connection_state", "connected")) == "handover_pending"
                )
            )
            handover_interruption_ue_count = int(
                sum(
                    1
                    for connection in connection_map.values()
                    if str(getattr(connection, "connection_state", "connected")) == "handover_interruption"
                )
            )
            handover_completed_ue_count = int(
                sum(
                    1
                    for connection in connection_map.values()
                    if getattr(connection, "last_handover_slot", None) is not None
                )
            )
        else:
            connected_ue_count = int(len(ue_list))
            out_of_sync_ue_count = 0
            handover_pending_ue_count = 0
            handover_interruption_ue_count = 0
            handover_completed_ue_count = 0
        beamforming_map = getattr(state, "beamforming", {})
        if not isinstance(beamforming_map, dict):
            beamforming_map = {}

        scheduler_state = getattr(state, "scheduler", None)
        scheduled_ue_ids = get_cluster_scheduled_ue_ids(state)
        scheduled_ue_ids_str = ";".join(str(item) for item in scheduled_ue_ids)
        grants = get_cluster_grants(state)
        grant_count = int(len(grants))
        retx_grant_count = int(sum(1 for grant in grants if bool(getattr(grant, "is_retx", False))))
        scheduled_logical_channel_count = int(
            sum(len(list(getattr(grant, "logical_channel_allocations", []) or [])) for grant in grants)
        )
        prioritized_logical_channel_bytes = int(
            sum(
                int(getattr(allocation, "prioritized_bytes", 0))
                for grant in grants
                for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
            )
        )
        best_effort_logical_channel_bytes = int(
            sum(
                int(getattr(allocation, "best_effort_bytes", 0))
                for grant in grants
                for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
            )
        )
        scheduled_drb_ids = sorted(
            {
                int(getattr(allocation, "drb_id", 0))
                for grant in grants
                for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                if getattr(allocation, "drb_id", None) is not None
            }
        )
        scheduled_lcg_ids = sorted(
            {
                int(getattr(allocation, "lcg_id", 0))
                for grant in grants
                for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                if getattr(allocation, "lcg_id", None) is not None
            }
        )
        scheduled_logical_channel_group_count = int(len(scheduled_lcg_ids))
        resource_grid_state = getattr(state, "resource_grid", None)
        cluster_resource_grid_summary = get_cluster_resource_grid_summary(state)
        managed_cell_count = int(len(getattr(state, "managed_cells", {}) or {}))
        if managed_cell_count > 1:
            allocated_prbs_total = int(
                sum(int(getattr(grant, "allocated_prbs", 0)) for grant in grants)
            )
            allocated_symbols_total = int(
                sum(int(getattr(grant, "allocated_symbols", 0)) for grant in grants)
            )
        elif resource_grid_state is not None and len(getattr(resource_grid_state, "allocations", [])) > 0:
            allocated_prbs_total = int(
                len(
                    {
                        int(prb_index)
                        for allocation in getattr(resource_grid_state, "allocations", [])
                        for prb_index in list(getattr(allocation, "resource_block_indices", []) or [])
                    }
                )
            )
            allocated_symbols_total = int(
                len(
                    {
                        int(symbol_index)
                        for allocation in getattr(resource_grid_state, "allocations", [])
                        for symbol_index in list(getattr(allocation, "allocated_symbol_indices", []) or [])
                    }
                )
            )
        else:
            allocated_prbs_total = int(sum(int(getattr(grant, "allocated_prbs", 0)) for grant in grants))
            allocated_symbols_total = int(sum(int(getattr(grant, "allocated_symbols", 0)) for grant in grants))
        if int(cluster_resource_grid_summary["n_prbs_total"]) > 0:
            allocated_prbs_total = min(
                int(allocated_prbs_total),
                int(cluster_resource_grid_summary["n_prbs_total"]),
            )
        grant_bytes_budget_total = int(sum(int(getattr(grant, "grant_bytes_budget", 0)) for grant in grants))
        grant_summary = ";".join(
            f"{getattr(grant, 'ue_id', 'unknown')}:{int(getattr(grant, 'allocated_prbs', 0))}prb/{int(getattr(grant, 'allocated_symbols', 0))}sym"
            f"@p{int(getattr(grant, 'start_prb', 0))}/s{int(getattr(grant, 'start_symbol', 0))}"
            f"{'-retx' if bool(getattr(grant, 'is_retx', False)) else ''}"
            for grant in grants
        )
        scheduled_layer_counts = [
            int(getattr(beamforming_map.get(str(ue_id)), "layer_count", 1))
            for ue_id in scheduled_ue_ids
        ]
        resource_grid_re_total = int(
            sum(int(getattr(item, "resource_grid_re_count", 0)) for item in getattr(phy_result, "ue_results", []))
        )
        scheduled_overlap_fractions = [
            float(getattr(item, "resource_grid_overlap_fraction", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_interference_penalties_db = [
            float(getattr(item, "interference_penalty_db", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_intra_cell_interference_penalties_db = [
            float(getattr(item, "intra_cell_interference_penalty_db", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_inter_cell_interference_penalties_db = [
            float(getattr(item, "inter_cell_interference_penalty_db", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_inter_cell_interference_overlap_fractions = [
            float(getattr(item, "inter_cell_interference_overlap_fraction", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_inter_cell_frequency_coupling_fractions = [
            float(getattr(item, "inter_cell_frequency_coupling_fraction", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_inter_cell_interference_source_counts = [
            float(getattr(item, "inter_cell_interference_source_count", 0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_spatial_compatibility_scores = [
            float(getattr(item, "spatial_compatibility_score", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        scheduled_spatial_multiplexing_gains_db = [
            float(getattr(item, "spatial_multiplexing_gain_db", 0.0))
            for item in getattr(phy_result, "ue_results", [])
            if bool(getattr(item, "is_scheduled", False))
        ]
        avg_layer_count_scheduled = (
            float(sum(scheduled_layer_counts) / float(len(scheduled_layer_counts)))
            if len(scheduled_layer_counts) > 0
            else None
        )

        ue_results = list(getattr(phy_result, "ue_results", [])) if phy_result is not None else []
        scheduled_results = [r for r in ue_results if bool(getattr(r, "is_scheduled", False))]
        new_tx_bytes = int(
            sum(
                int(getattr(r, "scheduled_bytes", getattr(r, "transport_block_size_bytes", 0)))
                for r in scheduled_results
                if not bool(getattr(r, "is_retx", False))
            )
        )
        retx_tx_bytes = int(
            sum(
                int(getattr(r, "scheduled_bytes", getattr(r, "transport_block_size_bytes", 0)))
                for r in scheduled_results
                if bool(getattr(r, "is_retx", False))
            )
        )
        delivered_new_bytes = int(
            sum(
                int(getattr(r, "delivered_bytes", getattr(r, "transport_block_size_bytes", 0)))
                for r in scheduled_results
                if not bool(getattr(r, "is_retx", False))
            )
        )
        delivered_retx_bytes = int(
            sum(
                int(getattr(r, "delivered_bytes", getattr(r, "transport_block_size_bytes", 0)))
                for r in scheduled_results
                if bool(getattr(r, "is_retx", False))
            )
        )
        grant_budget_utilization = (
            float(new_tx_bytes + retx_tx_bytes) / float(grant_bytes_budget_total)
            if grant_bytes_budget_total > 0 else 0.0
        )

        if scheduled_results:
            avg_effective_sinr_db = float(sum(float(getattr(r, "effective_sinr_db", getattr(r, "sinr_db", 0.0))) for r in scheduled_results) / len(scheduled_results))
            avg_pre_eq_sinr_db = float(sum(float(getattr(r, "pre_eq_sinr_db", getattr(r, "sinr_db", 0.0))) for r in scheduled_results) / len(scheduled_results))
            avg_post_eq_sinr_db = float(sum(float(getattr(r, "post_eq_sinr_db", getattr(r, "sinr_db", 0.0))) for r in scheduled_results) / len(scheduled_results))
            avg_beam_gain_db = float(sum(float(getattr(r, "beam_gain_db", 0.0)) for r in scheduled_results) / len(scheduled_results))
            avg_precoder_match_score = float(sum(float(getattr(r, "precoder_match_score", 1.0)) for r in scheduled_results) / len(scheduled_results))
            avg_spatial_compatibility_score = float(sum(float(getattr(r, "spatial_compatibility_score", 0.0)) for r in scheduled_results) / len(scheduled_results))
            avg_spatial_multiplexing_gain_db = float(sum(float(getattr(r, "spatial_multiplexing_gain_db", 0.0)) for r in scheduled_results) / len(scheduled_results))
            avg_mcs_index = float(sum(float(getattr(r, "mcs_index", 0)) for r in scheduled_results) / len(scheduled_results))
            ack_count = int(sum(1 for r in scheduled_results if bool(getattr(r, "ack", False))))
            nack_count = int(sum(1 for r in scheduled_results if not bool(getattr(r, "ack", False))))
        else:
            avg_effective_sinr_db = None
            avg_pre_eq_sinr_db = None
            avg_post_eq_sinr_db = None
            avg_beam_gain_db = None
            avg_precoder_match_score = None
            avg_spatial_compatibility_score = None
            avg_spatial_multiplexing_gain_db = None
            avg_mcs_index = None
            ack_count = 0
            nack_count = 0

        link_adaptation_reason_counts: dict[str, int] = {}
        confidence_values: list[float] = []
        for result_item in scheduled_results:
            reason = str(getattr(result_item, "link_adaptation_reason", "") or "")
            if reason != "":
                link_adaptation_reason_counts[reason] = int(
                    link_adaptation_reason_counts.get(reason, 0)
                ) + 1
            confidence = getattr(result_item, "channel_quality_confidence", None)
            if confidence is not None:
                confidence_values.append(float(confidence))
        grant_channel_quality_confidence_mean = (
            float(sum(confidence_values) / float(len(confidence_values)))
            if len(confidence_values) > 0
            else None
        )

        denom = ack_count + nack_count
        ack_rate = float(ack_count) / float(denom) if denom > 0 else 0.0

        scheduling_decision = getattr(scheduler_state, "last_decision", None) if scheduler_state is not None else None
        grant_prbs_total = int(getattr(scheduling_decision, "total_prbs", allocated_prbs_total))
        grant_symbols_total = int(getattr(scheduling_decision, "total_symbols", allocated_symbols_total))
        granted_ue_ids = [str(getattr(grant, "ue_id", "unknown")) for grant in grants]
        granted_ue_ids_str = ";".join(granted_ue_ids)
        avg_true_sinr_db_all = (
            float(sum(float(self._safe_get(ue, "sinr_db", 0.0)) for ue in ue_list) / float(len(ue_list)))
            if len(ue_list) > 0 else None
        )
        avg_radio_quality_db_all_true = (
            float(
                sum(float(self._safe_get(ue, "radio_quality_db", 0.0)) for ue in ue_list)
                / float(len(ue_list))
            )
            if len(ue_list) > 0 else None
        )
        traffic_arrival_bytes_by_ue = str(
            getattr(getattr(state, "traffic", None), "arrival_bytes_by_ue", {})
        )
        traffic_arrival_bytes_by_logical_channel = str(
            getattr(getattr(state, "traffic", None), "arrival_bytes_by_logical_channel", {})
        )
        traffic_arrival_bytes_by_logical_channel_group = str(
            getattr(
                getattr(state, "traffic", None),
                "arrival_bytes_by_logical_channel_group",
                {},
            )
        )
        logical_channel_group_bsr_reported_bytes_total = int(
            sum(
                int(getattr(logical_channel_group, "bsr_reported_bytes", 0))
                for logical_channel_group in all_logical_channel_groups
            )
        )
        logical_channel_group_bsr_pending_bytes_total = int(
            sum(
                int(getattr(logical_channel_group, "bsr_pending_bytes", 0))
                for logical_channel_group in all_logical_channel_groups
            )
        )
        logical_channel_group_bsr_trigger_count = int(
            sum(
                1
                for logical_channel_group in all_logical_channel_groups
                if bool(getattr(logical_channel_group, "bsr_triggered", False))
            )
        )
        logical_channel_group_bsr_age_slots_mean = self._mean_or_none(
            [
                getattr(logical_channel_group, "bsr_age_slots", None)
                for logical_channel_group in all_logical_channel_groups
            ]
        )

        observable_metrics = {
            "queue_bytes": None,
            "throughput_mbps": None,
            "sinr_db": None,
            "bler": None,
            "sinr_db_scheduled": None,
            "bler_scheduled": None,
            "sinr_db_all": None,
            "radio_quality_db_all": None,
            "scheduled_ue_ids": scheduled_ue_ids_str,
            "scheduled_ue_count": int(len(scheduled_ue_ids)),
            "connected_ue_count": None,
            "out_of_sync_ue_count": None,
            "inter_cell_interference_penalty_db_mean": None,
            "inter_cell_interference_source_count_mean": None,
        }
        if observation is not None:
            raw_sinr = getattr(observation, "avg_sinr_db", None)
            raw_bler = getattr(observation, "avg_bler", None)
            raw_sinr_scheduled = getattr(observation, "avg_sinr_db_scheduled", None)
            raw_bler_scheduled = getattr(observation, "avg_bler_scheduled", None)
            raw_sinr_all = getattr(observation, "avg_sinr_db_all", None)
            raw_radio_all = getattr(observation, "avg_radio_quality_db_all", None)

            observable_metrics.update(
                {
                    "queue_bytes": int(getattr(observation, "total_queue_bytes", 0)),
                    "throughput_mbps": float(getattr(observation, "total_throughput_mbps", 0.0)),
                    "sinr_db": (float(raw_sinr) if raw_sinr is not None else None),
                    "bler": (float(raw_bler) if raw_bler is not None else None),
                    "sinr_db_scheduled": (
                        float(raw_sinr_scheduled) if raw_sinr_scheduled is not None else None
                    ),
                    "bler_scheduled": (
                        float(raw_bler_scheduled) if raw_bler_scheduled is not None else None
                    ),
                    "sinr_db_all": (
                        float(raw_sinr_all) if raw_sinr_all is not None else None
                    ),
                    "radio_quality_db_all": (
                        float(raw_radio_all) if raw_radio_all is not None else None
                    ),
                    "scheduled_ue_ids": ";".join(
                        str(item) for item in getattr(observation, "scheduled_ue_ids", [])
                    ),
                    "scheduled_ue_count": int(
                        getattr(observation, "num_scheduled_ues", len(scheduled_ue_ids))
                    ),
                    "connected_ue_count": int(
                        getattr(observation, "connected_ue_count", connected_ue_count)
                    ),
                    "out_of_sync_ue_count": int(
                        getattr(observation, "out_of_sync_ue_count", out_of_sync_ue_count)
                    ),
                    "handover_pending_ue_count": int(
                        getattr(observation, "handover_pending_ue_count", handover_pending_ue_count)
                    ),
                    "handover_interruption_ue_count": int(
                        getattr(
                            observation,
                            "handover_interruption_ue_count",
                            handover_interruption_ue_count,
                        )
                    ),
                    "handover_completed_ue_count": int(
                        getattr(observation, "handover_completed_ue_count", handover_completed_ue_count)
                    ),
                    "inter_cell_interference_penalty_db_mean": (
                        float(getattr(observation, "inter_cell_interference_penalty_db_mean", 0.0))
                        if getattr(observation, "inter_cell_interference_penalty_db_mean", None) is not None
                        else None
                    ),
                    "inter_cell_frequency_coupling_fraction_mean": (
                        float(getattr(observation, "inter_cell_frequency_coupling_fraction_mean", 0.0))
                        if getattr(observation, "inter_cell_frequency_coupling_fraction_mean", None) is not None
                        else None
                    ),
                    "inter_cell_interference_source_count_mean": (
                        float(getattr(observation, "inter_cell_interference_source_count_mean", 0.0))
                        if getattr(observation, "inter_cell_interference_source_count_mean", None) is not None
                        else None
                    ),
                }
            )

        measurement_state = getattr(state, "measurement", None)
        measurement_trace = {
            "observation_slot_id": int(getattr(observation, "slot_id", slot_id_value))
            if observation is not None else slot_id_value,
            "observation_produced_at_slot": getattr(observation, "produced_at_slot", None)
            if observation is not None else None,
            "observation_delivered_at_slot": getattr(observation, "delivered_at_slot", None)
            if observation is not None else None,
            "produced_at_slot": getattr(observation, "measurement_produced_at_slot", None)
            if observation is not None else getattr(measurement_state, "produced_at_slot", None),
            "delivered_at_slot": getattr(observation, "measurement_delivered_at_slot", None)
            if observation is not None else getattr(measurement_state, "delivered_at_slot", None),
            "age_slots": getattr(observation, "measurement_age_slots", None)
            if observation is not None else getattr(measurement_state, "age_slots", None),
            "queue_delay_slots_applied": getattr(observation, "measurement_queue_delay_slots", None)
            if observation is not None else getattr(measurement_state, "queue_delay_slots", None),
            "queue_age_slots_mean": getattr(observation, "measurement_queue_age_slots_mean", None)
            if observation is not None else None,
            "sinr_window_slots": getattr(observation, "measurement_sinr_window_slots", None)
            if observation is not None else getattr(measurement_state, "sinr_window_slots", None),
            "sinr_noise_std_db_applied": getattr(
                observation,
                "measurement_sinr_noise_std_db_applied",
                None,
            ) if observation is not None else getattr(measurement_state, "sinr_noise_std_db_applied", None),
            "sinr_effective_sample_count_mean": getattr(
                observation,
                "measurement_sinr_effective_sample_count_mean",
                None,
            ) if observation is not None else None,
            "bler_window_tbs": getattr(observation, "measurement_bler_window_tbs", None)
            if observation is not None else getattr(measurement_state, "bler_window_tbs", None),
            "bler_sample_count_mean": getattr(
                observation,
                "measurement_bler_sample_count_mean",
                None,
            ) if observation is not None else None,
            "throughput_granularity_slots": getattr(
                observation,
                "measurement_throughput_granularity_slots",
                None,
            ) if observation is not None else getattr(measurement_state, "throughput_granularity_slots", None),
            "throughput_sample_count_mean": getattr(
                observation,
                "measurement_throughput_sample_count_mean",
                None,
            ) if observation is not None else None,
        }

        control_state = getattr(state, "control", None)
        decision_metadata = getattr(control_state, "decision_metadata", None)
        input_observation_trace = dict(
            getattr(control_state, "input_observation_trace", {})
        ) if control_state is not None else {}
        oran_lifecycle_resources = getattr(control_state, "oran_lifecycle_resources", None)
        a1_policy_instances = list(getattr(oran_lifecycle_resources, "a1_policy_instances", []) or [])
        a1_policy_statuses = list(getattr(oran_lifecycle_resources, "a1_policy_statuses", []) or [])
        a1_ei_jobs = list(getattr(oran_lifecycle_resources, "a1_ei_jobs", []) or [])
        e2_subscriptions = list(getattr(oran_lifecycle_resources, "e2_subscriptions", []) or [])
        e2_indications = list(getattr(oran_lifecycle_resources, "e2_indications", []) or [])
        e2_control_requests = list(getattr(oran_lifecycle_resources, "e2_control_requests", []) or [])
        e2_control_outcomes = list(getattr(oran_lifecycle_resources, "e2_control_outcomes", []) or [])
        rl_tuning_summary = self._rl_tuning_constraint_summary(control_state)
        effective_control_profile = self._effective_control_profile_summary(state)
        control_trace = {
            "decision_id": getattr(decision_metadata, "decision_id", None),
            "decision_source": getattr(decision_metadata, "source", None),
            "decision_issued_at_slot": getattr(decision_metadata, "issued_at_slot", None),
            "decision_reason": getattr(decision_metadata, "reason", None),
            "merged_config_action_count": int(
                len(getattr(control_state, "active_config_actions", []))
            ) if control_state is not None else 0,
            "merged_config_target_cell_ids": [
                str(target_cell_id)
                for target_cell_id in sorted(
                    {
                        str(target_cell_id)
                        for action in list(getattr(control_state, "active_config_actions", []) or [])
                        for target_cell_id in (
                            [getattr(action, "target_cell_id", None)]
                            if getattr(action, "target_cell_id", None) is not None
                            else []
                        )
                    }
                )
            ] if control_state is not None else [],
            "merged_config_apply_scopes": [
                str(scope_name)
                for scope_name in sorted(
                    {
                        str(getattr(action, "apply_scope", "cell"))
                        for action in list(getattr(control_state, "active_config_actions", []) or [])
                    }
                )
            ] if control_state is not None else [],
            "actuation_outcome_count": int(len(getattr(control_state, "actuation_outcomes", [])))
            if control_state is not None else 0,
            "actuation_target_cell_ids": [
                str(target_cell_id)
                for target_cell_id in sorted(
                    {
                        str(target_cell_id)
                        for outcome in list(getattr(control_state, "actuation_outcomes", []) or [])
                        for target_cell_id in list(outcome.get("target_cell_ids", []) or [])
                    }
                )
            ] if control_state is not None else [],
            "input_observation_slot_id": input_observation_trace.get("observation_slot_id"),
            "input_observation_produced_at_slot": input_observation_trace.get("observation_produced_at_slot"),
            "input_observation_delivered_at_slot": input_observation_trace.get("observation_delivered_at_slot"),
            "input_measurement_produced_at_slot": input_observation_trace.get("measurement_produced_at_slot"),
            "input_measurement_delivered_at_slot": input_observation_trace.get("measurement_delivered_at_slot"),
            "input_measurement_age_slots": input_observation_trace.get("measurement_age_slots"),
            "input_queue_delay_slots_applied": input_observation_trace.get("measurement_queue_delay_slots"),
            "input_sinr_window_slots": input_observation_trace.get("measurement_sinr_window_slots"),
            "input_sinr_noise_std_db_applied": input_observation_trace.get(
                "measurement_sinr_noise_std_db_applied"
            ),
            "input_bler_window_tbs": input_observation_trace.get("measurement_bler_window_tbs"),
            "input_throughput_granularity_slots": input_observation_trace.get(
                "measurement_throughput_granularity_slots"
            ),
            "input_queue_age_slots_mean": input_observation_trace.get(
                "measurement_queue_age_slots_mean"
            ),
            "input_sinr_effective_sample_count_mean": input_observation_trace.get(
                "measurement_sinr_effective_sample_count_mean"
            ),
            "input_bler_sample_count_mean": input_observation_trace.get(
                "measurement_bler_sample_count_mean"
            ),
            "input_throughput_sample_count_mean": input_observation_trace.get(
                "measurement_throughput_sample_count_mean"
            ),
            "a1_policy_type_count": int(
                len(list(getattr(oran_lifecycle_resources, "a1_policy_types", []) or []))
            ),
            "a1_policy_instance_count": int(len(a1_policy_instances)),
            "a1_policy_active_count": int(
                sum(1 for instance in a1_policy_instances if str(getattr(instance, "status", "")) == "active")
            ),
            "a1_ei_type_count": int(
                len(list(getattr(oran_lifecycle_resources, "a1_ei_types", []) or []))
            ),
            "a1_ei_job_count": int(len(a1_ei_jobs)),
            "a1_ei_active_job_count": int(
                sum(1 for job in a1_ei_jobs if str(getattr(job, "status", "")) == "active")
            ),
            "e2_subscription_count": int(len(e2_subscriptions)),
            "e2_active_subscription_count": int(
                sum(1 for subscription in e2_subscriptions if str(getattr(subscription, "status", "")) == "active")
            ),
            "e2_indication_count": int(len(e2_indications)),
            "e2_control_request_count": int(len(e2_control_requests)),
            "e2_control_outcome_count": int(len(e2_control_outcomes)),
            "e2_control_request_status_counts": {
                str(status_name): int(status_count)
                for status_name, status_count in sorted(
                    {
                        str(getattr(request, "status", "unknown")): sum(
                            1
                            for candidate in e2_control_requests
                            if str(getattr(candidate, "status", "unknown")) == str(getattr(request, "status", "unknown"))
                        )
                        for request in e2_control_requests
                    }.items(),
                    key=lambda item: item[0],
                )
            },
            "e2_control_outcome_status_counts": {
                str(status_name): int(status_count)
                for status_name, status_count in sorted(
                    {
                        str(getattr(outcome, "status", "unknown")): sum(
                            1
                            for candidate in e2_control_outcomes
                            if str(getattr(candidate, "status", "unknown")) == str(getattr(outcome, "status", "unknown"))
                        )
                        for outcome in e2_control_outcomes
                    }.items(),
                    key=lambda item: item[0],
                )
            },
            "effective_control_profile": effective_control_profile,
        }
        control_trace.update(rl_tuning_summary)
        scheduling_demands = get_cluster_scheduler_demands(state)
        scheduling_stage_trace = get_cluster_scheduler_stage_trace(state)
        scheduler_trace = {
            "demand_count": int(len(scheduling_demands)),
            "demand_true_backlog_bytes_total": int(
                sum(
                    int(getattr(demand, "true_backlog_bytes", getattr(demand, "backlog_bytes", 0)))
                    for demand in scheduling_demands
                )
            ),
            "demand_backlog_bytes_total": int(
                sum(int(getattr(demand, "backlog_bytes", 0)) for demand in scheduling_demands)
            ),
            "demand_visible_bytes_total": int(
                sum(
                    int(getattr(demand, "demand_bytes", getattr(demand, "backlog_bytes", 0)))
                    for demand in scheduling_demands
                )
            ),
            "demand_observed_backlog_bytes_total": int(
                sum(int(getattr(demand, "observed_backlog_bytes", 0)) for demand in scheduling_demands)
            ),
            "demand_bsr_like_bytes_total": int(
                sum(int(getattr(demand, "bsr_like_bytes", 0)) for demand in scheduling_demands)
            ),
            "demand_bsr_reported_bytes_total": int(
                sum(int(getattr(demand, "bsr_reported_bytes", 0)) for demand in scheduling_demands)
            ),
            "demand_bsr_pending_bytes_total": int(
                sum(int(getattr(demand, "bsr_pending_bytes", 0)) for demand in scheduling_demands)
            ),
            "demand_harq_pending_bytes_total": int(
                sum(int(getattr(demand, "harq_pending_bytes", 0)) for demand in scheduling_demands)
            ),
            "demand_hol_delay_ms_mean": self._mean_or_none(
                [float(getattr(demand, "hol_delay_ms", 0.0)) for demand in scheduling_demands]
            ),
            "channel_quality_confidence_mean": self._mean_or_none(
                [
                    float(getattr(demand, "channel_quality_confidence", 1.0))
                    for demand in scheduling_demands
                ]
            ),
            "logical_channel_demand_count": int(
                sum(
                    len(list(getattr(demand, "logical_channel_demands", []) or []))
                    for demand in scheduling_demands
                )
            ),
            "logical_channel_group_demand_count": int(
                sum(
                    len(list(getattr(demand, "logical_channel_group_demands", []) or []))
                    for demand in scheduling_demands
                )
            ),
            "cqi_validity_age_slots_mean": self._mean_or_none(
                [float(getattr(demand, "cqi_validity_age_slots", 0)) for demand in scheduling_demands]
            ),
            "ri_validity_age_slots_mean": self._mean_or_none(
                [float(getattr(demand, "ri_validity_age_slots", 0)) for demand in scheduling_demands]
            ),
            "pmi_validity_age_slots_mean": self._mean_or_none(
                [float(getattr(demand, "pmi_validity_age_slots", 0)) for demand in scheduling_demands]
            ),
            "managed_cell_count": int(len(getattr(state, "managed_cells", {}) or {})),
            "demand_source": getattr(scheduling_stage_trace, "demand_source", None),
            "demand_visible_count": int(
                len(getattr(scheduling_stage_trace, "demand_visible_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "low_confidence_channel_count": int(
                len(getattr(scheduling_stage_trace, "low_confidence_channel_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "stale_csi_count": int(
                len(getattr(scheduling_stage_trace, "stale_csi_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "harq_urgent_count": int(
                len(getattr(scheduling_stage_trace, "harq_urgent_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "feasible_candidate_count": int(
                len(getattr(scheduling_stage_trace, "feasible_candidate_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "retx_candidate_count": int(
                len(getattr(scheduling_stage_trace, "retx_candidate_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "new_tx_candidate_count": int(
                len(getattr(scheduling_stage_trace, "new_tx_candidate_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "selected_retx_count": int(
                len(getattr(scheduling_stage_trace, "selected_retx_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "selected_new_tx_count": int(
                len(getattr(scheduling_stage_trace, "selected_new_tx_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "granted_ue_count": int(
                len(getattr(scheduling_stage_trace, "granted_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "filtered_out_count": int(
                len(getattr(scheduling_stage_trace, "filtered_out_ue_ids", []) or [])
            ) if scheduling_stage_trace is not None else 0,
            "feasible_candidate_ue_ids": list(
                getattr(scheduling_stage_trace, "feasible_candidate_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "demand_visible_ue_ids": list(
                getattr(scheduling_stage_trace, "demand_visible_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "low_confidence_channel_ue_ids": list(
                getattr(scheduling_stage_trace, "low_confidence_channel_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "stale_csi_ue_ids": list(
                getattr(scheduling_stage_trace, "stale_csi_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "harq_urgent_ue_ids": list(
                getattr(scheduling_stage_trace, "harq_urgent_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "filtered_out_ue_ids": list(
                getattr(scheduling_stage_trace, "filtered_out_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "filtered_out_reasons": dict(
                getattr(scheduling_stage_trace, "filtered_out_reasons", {}) or {}
            ) if scheduling_stage_trace is not None else {},
            "retx_candidate_ue_ids": list(
                getattr(scheduling_stage_trace, "retx_candidate_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "new_tx_candidate_ue_ids": list(
                getattr(scheduling_stage_trace, "new_tx_candidate_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "selected_retx_ue_ids": list(
                getattr(scheduling_stage_trace, "selected_retx_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "selected_new_tx_ue_ids": list(
                getattr(scheduling_stage_trace, "selected_new_tx_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "grant_candidate_ue_ids": list(
                getattr(scheduling_stage_trace, "grant_candidate_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "granted_ue_ids": list(
                getattr(scheduling_stage_trace, "granted_ue_ids", []) or []
            ) if scheduling_stage_trace is not None else [],
            "selection_reasons_by_ue": dict(
                getattr(scheduling_stage_trace, "selection_reasons_by_ue", {}) or {}
            ) if scheduling_stage_trace is not None else {},
            "rejection_reasons_by_ue": dict(
                getattr(scheduling_stage_trace, "rejection_reasons_by_ue", {}) or {}
            ) if scheduling_stage_trace is not None else {},
            "resource_pressure_source_by_ue": dict(
                getattr(scheduling_stage_trace, "resource_pressure_source_by_ue", {}) or {}
            ) if scheduling_stage_trace is not None else {},
            "harq_preemption_causes_by_ue": dict(
                getattr(scheduling_stage_trace, "harq_preemption_causes_by_ue", {}) or {}
            ) if scheduling_stage_trace is not None else {},
            "constraint_hits_by_ue": dict(
                getattr(scheduling_stage_trace, "constraint_hits_by_ue", {}) or {}
            ) if scheduling_stage_trace is not None else {},
            "lc_multiplexing_summaries_by_ue": dict(
                getattr(scheduling_stage_trace, "lc_multiplexing_summaries_by_ue", {}) or {}
            ) if scheduling_stage_trace is not None else {},
            "harq_preemption_count": int(
                len(getattr(scheduling_stage_trace, "harq_preemption_causes_by_ue", {}) or {})
            ) if scheduling_stage_trace is not None else 0,
        }
        phy_comparison_trace = self._build_phy_comparison_trace(
            phy_result=phy_result,
            shadow_phy_result=shadow_phy_result,
            shadow_phy_label=shadow_phy_label,
        )

        true_metrics = {
            "queue_bytes": int(total_queue_bytes),
            "throughput_mbps": float(total_throughput_mbps),
            "avg_sinr_db_all": avg_true_sinr_db_all,
            "avg_radio_quality_db_all": avg_radio_quality_db_all_true,
            "avg_effective_sinr_db": avg_effective_sinr_db,
            "avg_pre_eq_sinr_db": avg_pre_eq_sinr_db,
            "avg_post_eq_sinr_db": avg_post_eq_sinr_db,
            "avg_beam_gain_db": avg_beam_gain_db,
            "avg_precoder_match_score": avg_precoder_match_score,
            "avg_spatial_compatibility_score": avg_spatial_compatibility_score,
            "avg_spatial_multiplexing_gain_db": avg_spatial_multiplexing_gain_db,
            "avg_mcs_index": avg_mcs_index,
            "tx_tb_count_this_slot": int(len(scheduled_results)),
            "ack_count_this_slot": ack_count,
            "nack_count_this_slot": nack_count,
            "ack_rate_this_slot": ack_rate,
            "new_tx_bytes_this_slot": new_tx_bytes,
            "retx_tx_bytes_this_slot": retx_tx_bytes,
            "delivered_new_bytes_this_slot": delivered_new_bytes,
            "delivered_retx_bytes_this_slot": delivered_retx_bytes,
            "grant_prbs_total": grant_prbs_total,
            "grant_prbs_used": allocated_prbs_total,
            "grant_symbols_total": grant_symbols_total,
            "grant_symbols_used": allocated_symbols_total,
            "grant_bytes_budget_total": grant_bytes_budget_total,
            "grant_budget_utilization": grant_budget_utilization,
            "grant_channel_quality_confidence_mean": grant_channel_quality_confidence_mean,
            "link_adaptation_reason_counts": link_adaptation_reason_counts,
            "phy_modes_scheduled": ";".join(phy_modes_scheduled),
            "waveform_backends_scheduled": ";".join(waveform_backends_scheduled),
            "avg_effective_data_re_count_scheduled": avg_effective_data_re_count_scheduled,
            "avg_dmrs_overhead_fraction_scheduled": avg_dmrs_overhead_fraction_scheduled,
            "granted_ue_ids": granted_ue_ids_str,
            "granted_ue_count": grant_count,
            "retx_grant_count": retx_grant_count,
            "logical_channel_count_total": total_logical_channel_count,
            "logical_channel_group_count_total": total_logical_channel_group_count,
            "scheduled_logical_channel_count": scheduled_logical_channel_count,
            "scheduled_logical_channel_group_count": scheduled_logical_channel_group_count,
            "prioritized_logical_channel_bytes": prioritized_logical_channel_bytes,
            "best_effort_logical_channel_bytes": best_effort_logical_channel_bytes,
            "logical_channel_group_bsr_reported_bytes_total": logical_channel_group_bsr_reported_bytes_total,
            "logical_channel_group_bsr_pending_bytes_total": logical_channel_group_bsr_pending_bytes_total,
            "logical_channel_group_bsr_trigger_count": logical_channel_group_bsr_trigger_count,
            "logical_channel_group_bsr_age_slots_mean": logical_channel_group_bsr_age_slots_mean,
            "scheduled_drb_count": int(len(scheduled_drb_ids)),
            "scheduled_drb_ids": ";".join(str(item) for item in scheduled_drb_ids),
            "scheduled_lcg_count": int(len(scheduled_lcg_ids)),
            "scheduled_lcg_ids": ";".join(str(item) for item in scheduled_lcg_ids),
            "scheduled_ue_ids": scheduled_ue_ids_str,
            "scheduled_ue_count": int(len(scheduled_ue_ids)),
            "connected_ue_count": connected_ue_count,
            "out_of_sync_ue_count": out_of_sync_ue_count,
            "handover_pending_ue_count": handover_pending_ue_count,
            "handover_interruption_ue_count": handover_interruption_ue_count,
            "handover_completed_ue_count": handover_completed_ue_count,
            "gnb_num_tx_ant": int(getattr(getattr(state, "antenna", None), "gnb_num_tx_ant", 1)),
            "carrier_frequency_hz": float(getattr(getattr(state, "antenna", None), "carrier_frequency_hz", 0.0)),
            "bandwidth_hz": float(getattr(getattr(state, "antenna", None), "bandwidth_hz", 0.0)),
            "subcarrier_spacing_hz": float(getattr(getattr(state, "antenna", None), "subcarrier_spacing_hz", 0.0)),
            "cyclic_prefix_type": str(getattr(getattr(state, "antenna", None), "cyclic_prefix_type", "normal")),
            "num_ofdm_symbols_per_slot": int(
                getattr(getattr(state, "antenna", None), "num_ofdm_symbols_per_slot", 0)
            ),
            "n_prbs_total": int(cluster_resource_grid_summary["n_prbs_total"]),
            "scheduled_layer_count_total": int(cluster_resource_grid_summary["scheduled_layer_count_total"]),
            "avg_layer_count_scheduled": avg_layer_count_scheduled,
            "resource_grid_total_re": int(cluster_resource_grid_summary["total_resource_elements"]),
            "resource_grid_re_total": resource_grid_re_total,
            "resource_grid_utilization": float(cluster_resource_grid_summary["utilization"]),
            "resource_grid_overlap_fraction_mean": self._mean_or_none(scheduled_overlap_fractions),
            "interference_penalty_db_mean": self._mean_or_none(scheduled_interference_penalties_db),
            "intra_cell_interference_penalty_db_mean": self._mean_or_none(
                scheduled_intra_cell_interference_penalties_db
            ),
            "inter_cell_interference_penalty_db_mean": self._mean_or_none(
                scheduled_inter_cell_interference_penalties_db
            ),
            "inter_cell_interference_overlap_fraction_mean": self._mean_or_none(
                scheduled_inter_cell_interference_overlap_fractions
            ),
            "inter_cell_frequency_coupling_fraction_mean": self._mean_or_none(
                scheduled_inter_cell_frequency_coupling_fractions
            ),
            "inter_cell_interference_source_count_mean": self._mean_or_none(
                scheduled_inter_cell_interference_source_counts
            ),
            "inter_cell_interference_source_count_total": int(
                sum(int(item) for item in scheduled_inter_cell_interference_source_counts)
            ),
            "spatial_compatibility_score_mean": self._mean_or_none(scheduled_spatial_compatibility_scores),
            "spatial_multiplexing_gain_db_mean": self._mean_or_none(scheduled_spatial_multiplexing_gains_db),
            "mu_mimo_overlap_pair_count": int(cluster_resource_grid_summary["mu_mimo_overlap_pair_count"]),
            "mu_mimo_group_count": int(cluster_resource_grid_summary["mu_mimo_group_count"]),
            "mu_mimo_ue_count": int(cluster_resource_grid_summary["mu_mimo_ue_count"]),
            "adjacent_interference_pair_count": int(
                cluster_resource_grid_summary["adjacent_interference_pair_count"]
            ),
            "interference_group_count": int(cluster_resource_grid_summary["interference_group_count"]),
            "managed_cell_count": managed_cell_count,
            "scheduler_mode": str(getattr(getattr(state, "scheduler", None), "scheduler_mode", "unknown")),
            "scheduler_packing_strategy": str(
                getattr(getattr(state, "scheduler", None), "packing_strategy", "frequency_first")
            ),
            "tx_power_dbm": float(getattr(getattr(state, "cell", None), "tx_power_dbm", 0.0)),
            "current_codebook_id": int(getattr(getattr(state, "cell", None), "current_codebook_id", 0)),
            "current_beam_profile_id": int(getattr(getattr(state, "cell", None), "current_beam_profile_id", 0)),
            "total_pending_retx_bytes": int(total_pending_retx_bytes),
            "harq_flush_count_total": int(harq_flush_count_total),
            "active_harq_process_count_total": int(active_harq_process_count_total),
            "avg_harq_retransmission_round": self._mean_or_none(
                [float(item) for item in harq_retransmission_rounds]
            ),
            "traffic_arrival_bytes_total": int(
                getattr(getattr(state, "traffic", None), "arrival_bytes_total", 0)
            ),
            "traffic_arrival_bytes_by_ue": traffic_arrival_bytes_by_ue,
            "traffic_arrival_bytes_by_logical_channel": traffic_arrival_bytes_by_logical_channel,
            "traffic_arrival_bytes_by_logical_channel_group": traffic_arrival_bytes_by_logical_channel_group,
            "grant_summary": grant_summary,
        }

        metric_item = {
            "slot_id": slot_id_value,
            "sim_time_ms": sim_time_ms,
            "true_metrics": true_metrics,
            "observable_metrics": observable_metrics,
            "measurement_trace": measurement_trace,
            "control_trace": control_trace,
            "scheduler_trace": scheduler_trace,
            "phy_comparison_trace": phy_comparison_trace,
        }

        slot_records.append(metric_item)

    def update(self, state, slot_id: int, phy_result) -> None:
        """
        Convenience method to perform all state update steps for one slot.
        INPUT
        state: Simulator state object.
        slot_id: Current slot id.
        phy_result: PHY result object.
        OUTPUT
        None.
        """
        self.apply_phy_result(state, phy_result)
        self.record_snapshot(state)
        self.record_slot_metrics(state, slot_id=slot_id, phy_result=phy_result)



