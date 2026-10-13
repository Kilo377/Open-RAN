"""
Grant-based NR downlink MAC scheduler.

The scheduler approximates an O-DU High / 3GPP NR MAC scheduling loop that
builds DL grants over PRB and OFDM-symbol resources before handing them to
the abstract PHY / PDSCH execution path.
"""

from __future__ import annotations

from typing import Any, List

from core.debug import debug_print
from models.scheduling import (
    LogicalChannelDemand,
    LogicalChannelGroupDemand,
    LogicalChannelGrant,
    SchedulingDecision,
    SchedulingStageTrace,
    UeGrant,
    UeSchedulingDemand,
)
from phy.mcs_profiles import (
    estimate_tbs_bytes,
    normalize_mcs_table_id,
    normalize_tbs_selection_method,
)
from ran.scheduler_base import SchedulerBase
from state.harq_state import HarqProcessState
from state.logical_channel_group_state import LogicalChannelGroupState


class SimpleScheduler(SchedulerBase):
    _HARQ_RV_SEQUENCE = (0, 2, 3, 1)

    """
    Implements a minimal multi-mode grant-based scheduler.
    INPUT
    total_prbs_per_slot: The slot-level PRB budget.
    total_symbols_per_slot: The slot-level OFDM symbol budget.
    max_ues_per_slot: The maximum number of UEs that can receive grants.
    min_prbs_per_grant: The minimum PRBs reserved for one active grant.
    min_symbols_per_grant: The minimum OFDM symbols reserved for one active grant.
    retx_weight_bonus: Extra weight used when allocating resources to retransmissions.
    OUTPUT
    A scheduler object that updates the scheduler state for each slot.
    EXAMPLE OUTPUT
    scheduler = SimpleScheduler()
    scheduler.run(state)
    """

    def __init__(
        self,
        *,
        total_prbs_per_slot: int = 48,
        total_symbols_per_slot: int = 14,
        max_ues_per_slot: int = 2,
        min_prbs_per_grant: int = 4,
        min_symbols_per_grant: int = 4,
        retx_weight_bonus: float = 1000.0,
        packing_strategy: str = "frequency_first",
        enable_spatial_multiplexing: bool = False,
        max_mu_users_per_group: int = 2,
        min_mu_beam_separation_deg: float = 45.0,
        max_total_spatial_layers: int | None = None,
    ) -> None:
        self.total_prbs_per_slot = max(1, int(total_prbs_per_slot))
        self.total_symbols_per_slot = max(1, int(total_symbols_per_slot))
        self.max_ues_per_slot = max(1, int(max_ues_per_slot))
        self.min_prbs_per_grant = max(1, int(min_prbs_per_grant))
        self.min_symbols_per_grant = max(1, int(min_symbols_per_grant))
        self.retx_weight_bonus = float(retx_weight_bonus)
        self.packing_strategy = self._normalize_packing_strategy(packing_strategy)
        self.enable_spatial_multiplexing = bool(enable_spatial_multiplexing)
        self.max_mu_users_per_group = max(2, int(max_mu_users_per_group))
        self.min_mu_beam_separation_deg = max(0.0, float(min_mu_beam_separation_deg))
        if max_total_spatial_layers is None or str(max_total_spatial_layers).strip() == "":
            self.max_total_spatial_layers = None
        else:
            self.max_total_spatial_layers = max(1, int(max_total_spatial_layers))

    def _normalize_packing_strategy(self, packing_strategy: str | None) -> str:
        strategy = str(packing_strategy or "frequency_first").strip().lower()
        if strategy not in {"frequency_first", "time_first"}:
            return "frequency_first"
        return strategy

    def _get_packing_strategy(self, state: Any) -> str:
        scheduler_state = getattr(state, "scheduler", None)
        requested = getattr(scheduler_state, "packing_strategy", self.packing_strategy)
        return self._normalize_packing_strategy(requested)

    def _get_spatial_multiplexing_enabled(self, state: Any) -> bool:
        scheduler_state = getattr(state, "scheduler", None)
        requested = getattr(scheduler_state, "enable_spatial_multiplexing", self.enable_spatial_multiplexing)
        return bool(requested)

    def _get_max_mu_users_per_group(self, state: Any) -> int:
        scheduler_state = getattr(state, "scheduler", None)
        requested = getattr(scheduler_state, "max_mu_users_per_group", self.max_mu_users_per_group)
        return max(2, int(requested))

    def _get_min_mu_beam_separation_deg(self, state: Any) -> float:
        scheduler_state = getattr(state, "scheduler", None)
        requested = getattr(
            scheduler_state,
            "min_mu_beam_separation_deg",
            self.min_mu_beam_separation_deg,
        )
        return max(0.0, float(requested))

    def _get_max_total_spatial_layers(self, state: Any) -> int:
        scheduler_state = getattr(state, "scheduler", None)
        requested = getattr(scheduler_state, "max_total_spatial_layers", self.max_total_spatial_layers)
        if requested is not None and str(requested).strip() != "":
            return max(1, int(requested))
        antenna_state = getattr(state, "antenna", None)
        return max(1, int(getattr(antenna_state, "gnb_num_tx_ant", 1)))

    def _clear_scheduler_outputs(
        self,
        state: Any,
        *,
        demands: list[UeSchedulingDemand] | None = None,
        stage_trace: SchedulingStageTrace | None = None,
    ) -> None:
        state.scheduler.scheduled_ue_ids = []
        state.scheduler.grants = []
        try:
            last_demands = list(demands) if demands is not None else []
            last_stage_trace = stage_trace
            if last_stage_trace is None:
                last_stage_trace = SchedulingStageTrace(
                    demand_source="mac_state",
                    feasible_candidate_ue_ids=[],
                    filtered_out_ue_ids=[
                        str(getattr(ue, "ue_id", "unknown_ue"))
                        for ue in getattr(state, "ues", [])
                    ],
                    filtered_out_reasons={
                        str(getattr(ue, "ue_id", "unknown_ue")): "no_scheduler_output"
                        for ue in getattr(state, "ues", [])
                    },
                )
            state.scheduler.last_demands = last_demands
            state.scheduler.last_stage_trace = last_stage_trace
        except Exception:
            pass
        try:
            state.scheduler.packing_strategy = self._get_packing_strategy(state)
            state.scheduler.enable_spatial_multiplexing = self._get_spatial_multiplexing_enabled(state)
            state.scheduler.max_mu_users_per_group = self._get_max_mu_users_per_group(state)
            state.scheduler.min_mu_beam_separation_deg = self._get_min_mu_beam_separation_deg(state)
            state.scheduler.max_total_spatial_layers = self._get_max_total_spatial_layers(state)
        except Exception:
            pass
        state.scheduler.last_decision = SchedulingDecision(
            slot_id=int(getattr(state, "slot_id", -1)),
            scheduler_mode=str(getattr(state.scheduler, "scheduler_mode", "round_robin")),
            total_prbs=int(self.total_prbs_per_slot),
            total_symbols=int(self.total_symbols_per_slot),
            grants=[],
            demands=list(getattr(state.scheduler, "last_demands", []) or []),
            stage_trace=getattr(state.scheduler, "last_stage_trace", None),
            unallocated_prbs=int(self.total_prbs_per_slot),
            unallocated_symbols=int(self.total_symbols_per_slot),
        )

        for ue in getattr(state, "ues", []):
            try:
                ue.is_scheduled = False
                ue.allocated_prbs = 0
                ue.allocated_symbols = 0
                ue.start_prb = 0
                ue.start_symbol = 0
                ue.is_retx = False
                ue.priority_score = 0.0
                ue.grant_bytes_budget = 0
                ue.grant_target_mcs = None
                ue.spatial_group_id = None
                ue.spatial_group_size = 1
                ue.spatial_compatibility_score = 0.0
                ue.spatial_multiplexing_gain_db = 0.0
                ue.spatial_mode = "su_siso"
                ue.grant_selection_reason = ""
                ue.grant_rejection_reason = ""
                ue.grant_resource_pressure_source = ""
                ue.grant_harq_preemption_cause = None
                ue.grant_constraint_hits = []
                ue.grant_lc_multiplexing_summary = {}
                ue.scheduled_logical_channel_ids = []
                ue.scheduled_drb_ids = []
                ue.scheduled_lcg_ids = []
            except Exception:
                pass

    def _get_connection(self, state: Any, ue_id: str) -> Any | None:
        connections = getattr(state, "connections", {})
        if isinstance(connections, dict):
            return connections.get(str(ue_id))
        return None

    def _get_ue_by_id(self, state: Any, ue_id: str) -> Any | None:
        for ue in getattr(state, "ues", []):
            if str(getattr(ue, "ue_id", "")) == str(ue_id):
                return ue
        return None

    def _is_ue_connected(self, state: Any, ue: Any) -> bool:
        ue_id = str(getattr(ue, "ue_id", ""))
        connection = self._get_connection(state, ue_id)
        if connection is None:
            return True

        serving_cell_id = str(
            getattr(connection, "serving_cell_id", getattr(getattr(state, "cell", None), "cell_id", ""))
        )
        current_cell_id = str(getattr(getattr(state, "cell", None), "cell_id", ""))
        managed_cell_ids = list(getattr(getattr(state, "cell", None), "managed_cell_ids", []) or [])
        if len(managed_cell_ids) == 0:
            managed_cell_ids = [current_cell_id]
        connection_state = str(getattr(connection, "connection_state", "connected"))
        return connection_state == "connected" and serving_cell_id in managed_cell_ids

    def _get_eligible_ues(self, state: Any) -> list:
        if str(getattr(getattr(state, "cell", None), "cell_sleep_mode", "active")).strip().lower() == "deep_sleep":
            return []
        eligible_ues = []
        for ue in getattr(state, "ues", []):
            if not self._is_ue_connected(state, ue):
                continue
            queue_bytes = int(getattr(ue, "queue_bytes", 0))
            pending_retx_bytes = self._get_pending_retx_bytes(ue)
            if queue_bytes > 0 or pending_retx_bytes > 0:
                eligible_ues.append(ue)
        return eligible_ues

    def _get_retx_ues(self, eligible_ues: list) -> list:
        return [ue for ue in eligible_ues if self._get_pending_retx_bytes(ue) > 0]

    def _get_current_slot_id(self, state: Any) -> int:
        return int(getattr(state, "slot_id", 0) or 0)

    def _get_pending_retx_bytes(self, ue: Any, *, slot_id: int | None = None) -> int:
        active_process = self._get_active_harq_process(ue, slot_id=slot_id)
        if active_process is not None:
            return int(getattr(active_process, "pending_bytes", 0))
        return int(getattr(getattr(ue, "harq", None), "pending_retx_bytes", 0))

    def _get_harq_process_pool_size(self, harq: Any) -> int:
        return max(1, int(getattr(harq, "process_pool_size", 4) or 4))

    def _get_harq_processes(self, harq: Any) -> list[Any]:
        return list(getattr(harq, "processes", []) or [])

    def _set_harq_processes(self, harq: Any, processes: list[Any]) -> None:
        try:
            harq.processes = list(processes)
        except Exception:
            pass

    def _get_harq_process_by_id(self, ue: Any, process_id: int | None) -> Any | None:
        if process_id is None:
            return None
        harq = getattr(ue, "harq", None)
        if harq is None:
            return None
        for process in self._get_harq_processes(harq):
            if int(getattr(process, "process_id", -1)) == int(process_id):
                return process
        return None

    def _get_harq_rv_sequence(self, process: Any | None) -> list[int]:
        sequence = list(getattr(process, "rv_sequence", []) or [])
        normalized_sequence = [int(item) for item in sequence if str(item).strip() != ""]
        if len(normalized_sequence) == 0:
            return list(self._HARQ_RV_SEQUENCE)
        return normalized_sequence

    def _is_harq_process_busy(self, process: Any) -> bool:
        return bool(
            getattr(process, "is_busy", False)
            or bool(getattr(process, "is_waiting_ack", False))
            or int(getattr(process, "pending_bytes", 0)) > 0
        )

    def _is_harq_process_free(self, process: Any | None) -> bool:
        if process is None:
            return True
        return not self._is_harq_process_busy(process)

    def _record_harq_process_flush(self, harq: Any, *, reason: str, slot_id: int, process=None) -> None:
        try:
            harq.harq_flush_count = int(getattr(harq, "harq_flush_count", 0)) + 1
            harq.last_flush_reason = str(reason)
            if hasattr(harq, 'events'):
                harq.events.append(dict(kind='payload_flush', slot_id=slot_id, reason=reason,
                    process_id=getattr(process, 'process_id', None), tb_id=getattr(process, 'tb_id', None),
                    pending_bytes=int(getattr(process, 'pending_bytes', 0))))
            if reason in {"timeout", "retx_deadline"}:
                harq.last_timeout_slot = int(slot_id)
        except Exception:
            pass

    def _is_harq_process_expired(self, process: Any, *, slot_id: int) -> bool:
        if int(getattr(process, "pending_bytes", 0)) <= 0:
            return False
        timeout_slot = getattr(process, "timeout_slot", None)
        if timeout_slot is not None and int(slot_id) > int(timeout_slot):
            return True
        retx_deadline_slot = getattr(process, "retx_deadline_slot", None)
        if retx_deadline_slot is not None and int(slot_id) > int(retx_deadline_slot):
            return True
        return False

    def _prune_expired_harq_processes(self, state: Any, ue: Any) -> None:
        harq = getattr(ue, "harq", None)
        if harq is None:
            return
        current_slot = self._get_current_slot_id(state)
        retained_processes: list[Any] = []
        flush_reason = None
        for process in self._get_harq_processes(harq):
            if int(getattr(process, 'pending_bytes', 0)) <= 0:
                harq.empty_process_cleanup_count = int(getattr(harq, 'empty_process_cleanup_count', 0)) + 1
                if hasattr(harq, 'events'):
                    harq.events.append(dict(kind='empty_process_cleanup', slot_id=current_slot, process_id=process.process_id, pending_bytes=0))
                continue
            timeout_slot = getattr(process, "timeout_slot", None)
            retx_deadline_slot = getattr(process, "retx_deadline_slot", None)
            if timeout_slot is not None and int(current_slot) > int(timeout_slot):
                flush_reason = "timeout"
                self._record_harq_process_flush(harq, reason=flush_reason, slot_id=current_slot, process=process)
                continue
            if retx_deadline_slot is not None and int(current_slot) > int(retx_deadline_slot):
                flush_reason = "retx_deadline"
                self._record_harq_process_flush(harq, reason=flush_reason, slot_id=current_slot, process=process)
                continue
            retained_processes.append(process)
        if flush_reason is None and len(retained_processes) == len(self._get_harq_processes(harq)):
            return
        self._set_harq_processes(harq, retained_processes)
        try:
            harq.pending_retx_bytes = int(
                sum(int(getattr(process, "pending_bytes", 0)) for process in retained_processes)
            )
            harq.pending_retx_logical_channels = list(
                getattr(retained_processes[0], "pending_logical_channels", []) or []
            ) if len(retained_processes) > 0 else []
            harq.retransmission_count = int(
                max(
                    [int(getattr(process, "retransmission_round", 0)) for process in retained_processes],
                    default=0,
                )
            )
            if flush_reason is not None:
                harq.last_decision = str(flush_reason).upper()
        except Exception:
            pass

    def _get_active_harq_process(self, ue: Any, *, slot_id: int | None = None) -> Any | None:
        harq = getattr(ue, "harq", None)
        if harq is None:
            return None
        processes = self._get_harq_processes(harq)
        if len(processes) > 0:
            active_processes = [
                process
                for process in processes
                if int(getattr(process, "pending_bytes", 0)) > 0
                and not (
                    slot_id is not None
                    and self._is_harq_process_expired(process, slot_id=int(slot_id))
                )
            ]
            if len(active_processes) > 0:
                return sorted(
                    active_processes,
                    key=lambda process: (
                        int(getattr(process, "retx_deadline_slot", 10**9) or 10**9),
                        int(getattr(process, "timeout_slot", 10**9) or 10**9),
                        -int(getattr(process, "retransmission_round", 0)),
                        int(getattr(process, "process_id", 0)),
                    ),
                )[0]

        pending_retx_bytes = int(getattr(harq, "pending_retx_bytes", 0))
        if pending_retx_bytes <= 0:
            return None

        synthesized_process_id = int(getattr(harq, "process_id", 0))
        synthesized_round = int(getattr(harq, "retransmission_count", 0))
        synthesized_process = HarqProcessState(
            process_id=synthesized_process_id,
            is_busy=True,
            ndi=int(getattr(harq, "ndi", 0)),
            redundancy_version=int(
                getattr(
                    harq,
                    "redundancy_version",
                    self._HARQ_RV_SEQUENCE[min(synthesized_round, len(self._HARQ_RV_SEQUENCE) - 1)],
                )
            ),
            retransmission_round=synthesized_round,
            pending_bytes=pending_retx_bytes,
            last_tx_bytes=int(getattr(harq, "last_tx_bytes", 0)),
            last_delivered_bytes=int(getattr(harq, "last_delivered_bytes", 0)),
            is_waiting_ack=bool(getattr(harq, "is_waiting_ack", False)),
            max_retransmission_rounds=int(getattr(harq, "max_retransmission_rounds", 4) or 4),
            flush_reason=getattr(harq, "last_flush_reason", None),
            pending_logical_channels=list(getattr(harq, "pending_retx_logical_channels", []) or []),
        )
        self._set_harq_processes(harq, [synthesized_process])
        return synthesized_process

    def _derive_harq_service_type(self, ue: Any, demand: UeSchedulingDemand | None = None) -> str:
        if demand is not None and str(getattr(demand, "traffic_type", "")).strip() != "":
            return str(getattr(demand, "traffic_type", "")).strip().lower()
        return str(getattr(ue, "traffic_type", "")).strip().lower()

    def _derive_harq_max_rounds(self, ue: Any, demand: UeSchedulingDemand | None = None) -> int:
        traffic_type = self._derive_harq_service_type(ue, demand)
        latency_budget_ms = float(getattr(ue, "latency_budget_ms", 50.0) or 50.0)
        if traffic_type in {"control", "urllc", "industrial_control"} or latency_budget_ms <= 10.0:
            return 2
        if traffic_type in {"voice", "phone", "ims_voice"}:
            return 3
        if traffic_type in {"iot", "mmtc", "telemetry"}:
            return 2
        return 4

    def _derive_harq_deadline_slot(
        self,
        state: Any,
        ue: Any,
        demand: UeSchedulingDemand | None = None,
        *,
        max_retransmission_rounds: int,
    ) -> int:
        slot_duration_ms = max(float(getattr(state, "slot_duration_ms", 1.0) or 1.0), 0.1)
        latency_budget_ms = float(getattr(ue, "latency_budget_ms", 50.0) or 50.0)
        budget_slots = max(1, int(latency_budget_ms / slot_duration_ms))
        return int(self._get_current_slot_id(state) + max(1, min(budget_slots, max_retransmission_rounds + 1)))

    def _derive_harq_timeout_slot(
        self,
        state: Any,
        *,
        retx_deadline_slot: int,
        max_retransmission_rounds: int,
    ) -> int:
        current_slot = self._get_current_slot_id(state)
        return int(max(retx_deadline_slot, current_slot + max(1, int(max_retransmission_rounds)) + 1))

    def _build_harq_tb_id(self, state: Any, ue_id: str, process_id: int, ndi: int) -> str:
        return f"{ue_id}:slot{self._get_current_slot_id(state)}:proc{int(process_id)}:ndi{int(ndi)}"

    def _get_free_harq_process_id(self, harq: Any) -> int | None:
        process_pool_size = self._get_harq_process_pool_size(harq)
        next_process_id = int(getattr(harq, "next_new_tx_process_id", 0) or 0) % process_pool_size
        processes_by_id = {
            int(getattr(process, "process_id", -1)): process
            for process in self._get_harq_processes(harq)
        }
        for offset in range(process_pool_size):
            candidate_process_id = int((next_process_id + offset) % process_pool_size)
            if self._is_harq_process_free(processes_by_id.get(candidate_process_id)):
                return candidate_process_id
        return None

    def _has_available_harq_process(self, harq: Any) -> bool:
        return self._get_free_harq_process_id(harq) is not None

    def _reserve_new_tx_harq_context(
        self,
        state: Any,
        ue: Any,
        demand: UeSchedulingDemand | None = None,
    ) -> tuple[int, int, int, int]:
        harq = getattr(ue, "harq", None)
        if harq is None:
            return 0, 0, 0, 0

        process_pool_size = self._get_harq_process_pool_size(harq)
        next_process_id = self._get_free_harq_process_id(harq)
        if next_process_id is None:
            next_process_id = int(getattr(harq, "next_new_tx_process_id", 0) or 0) % process_pool_size
        ndi_by_process_id = getattr(harq, "ndi_by_process_id", None)
        if not isinstance(ndi_by_process_id, dict):
            ndi_by_process_id = {}
        previous_ndi = int(ndi_by_process_id.get(int(next_process_id), 0))
        next_ndi = 1 - previous_ndi
        ndi_by_process_id[int(next_process_id)] = int(next_ndi)
        active_processes = self._get_harq_processes(harq)
        existing_process = None
        for process in active_processes:
            if int(getattr(process, "process_id", -1)) == int(next_process_id):
                existing_process = process
                break
        max_retransmission_rounds = self._derive_harq_max_rounds(ue, demand)
        retx_deadline_slot = self._derive_harq_deadline_slot(
            state,
            ue,
            demand,
            max_retransmission_rounds=max_retransmission_rounds,
        )
        timeout_slot = self._derive_harq_timeout_slot(
            state,
            retx_deadline_slot=retx_deadline_slot,
            max_retransmission_rounds=max_retransmission_rounds,
        )
        reserved_process = HarqProcessState(
            process_id=int(next_process_id),
            is_busy=True,
            ndi=int(next_ndi),
            tb_id=self._build_harq_tb_id(state, str(getattr(ue, "ue_id", "unknown_ue")), int(next_process_id), int(next_ndi)),
            redundancy_version=0,
            retransmission_round=0,
            pending_bytes=0,
            last_tx_bytes=int(getattr(existing_process, "last_tx_bytes", 0)) if existing_process is not None else 0,
            last_delivered_bytes=0,
            is_waiting_ack=True,
            rv_sequence=self._get_harq_rv_sequence(existing_process),
            retx_deadline_slot=int(retx_deadline_slot),
            timeout_slot=int(timeout_slot),
            flush_reason=None,
            service_type=self._derive_harq_service_type(ue, demand),
            max_retransmission_rounds=int(max_retransmission_rounds),
            reserved_at_slot=int(self._get_current_slot_id(state)),
            last_tx_slot=int(self._get_current_slot_id(state)),
            pending_logical_channels=[],
        )
        active_processes = [
            reserved_process
            if int(getattr(process, "process_id", -1)) == int(next_process_id)
            else process
            for process in active_processes
        ]
        if existing_process is None:
            active_processes.append(reserved_process)
        try:
            harq.ndi_by_process_id = dict(ndi_by_process_id)
            harq.next_new_tx_process_id = int((next_process_id + 1) % process_pool_size)
            harq.process_id = int(next_process_id)
            harq.ndi = int(next_ndi)
            harq.redundancy_version = 0
            harq.last_retransmission_round = 0
            harq.last_harq_process_id = int(next_process_id)
            harq.max_retransmission_rounds = int(max_retransmission_rounds)
            harq.last_flush_reason = None
        except Exception:
            pass
        self._set_harq_processes(harq, active_processes)
        return int(next_process_id), 0, int(next_ndi), 0

    def _get_measurement_for_ue(self, state: Any, ue_id: str) -> Any | None:
        measurement_state = getattr(state, "measurement", None)
        ue_measurements = getattr(measurement_state, "ue_measurements", None)
        if not isinstance(ue_measurements, dict):
            return None
        return ue_measurements.get(str(ue_id))

    def _get_link_state(self, state: Any, ue_id: str) -> Any | None:
        links = getattr(state, "links", {})
        if isinstance(links, dict):
            return links.get(str(ue_id))
        return None

    def _derive_qos_class(
        self,
        *,
        slice_id: str,
        traffic_type: str,
        priority: int,
    ) -> str:
        normalized_slice_id = str(slice_id).strip().lower()
        normalized_traffic_type = str(traffic_type).strip().lower()
        if normalized_slice_id != "":
            return normalized_slice_id
        if normalized_traffic_type != "":
            return normalized_traffic_type
        return f"priority_{max(0, int(priority))}"

    def _derive_five_qi(
        self,
        *,
        slice_id: str,
        traffic_type: str,
        priority: int,
    ) -> int | None:
        normalized_slice_id = str(slice_id).strip().lower()
        normalized_traffic_type = str(traffic_type).strip().lower()
        if normalized_traffic_type in {"control", "urllc", "industrial_control"}:
            return 80
        if normalized_traffic_type in {"voice", "phone", "ims_voice"}:
            return 1
        if normalized_traffic_type in {"video", "broadband", "embb"}:
            return 9
        if normalized_traffic_type in {"iot", "mmtc", "telemetry"}:
            return 70
        if normalized_slice_id == "urllc":
            return 80
        if normalized_slice_id == "embb":
            return 9
        if normalized_slice_id == "mmtc":
            return 70
        if int(priority) >= 3:
            return 80
        if int(priority) <= 0:
            return 9
        return None

    def _estimate_hol_delay_ms(self, state: Any, ue: Any, ue_measurement: Any | None) -> float:
        if getattr(ue, 'packet_ledger', None) is not None:
            measured = getattr(ue_measurement, 'hol_delay_ms', None)
            return float(measured if measured is not None else ue.hol_delay_ms)
        slot_duration_ms = max(float(getattr(state, "slot_duration_ms", 1.0)), 0.0)
        if ue_measurement is None:
            return min(
                float(getattr(ue, "latency_budget_ms", 0.0)),
                slot_duration_ms if int(getattr(ue, "queue_bytes", 0)) > 0 else 0.0,
            )
        queue_age_slots = int(
            getattr(
                ue_measurement,
                "queue_age_slots",
                getattr(ue_measurement, "age_slots", 0) or 0,
            )
            or 0
        )
        return max(0.0, float(queue_age_slots) * float(slot_duration_ms))

    def _estimate_csi_validity_age_slots(self, state: Any, ue_id: str, ue_measurement: Any | None) -> int:
        if ue_measurement is not None:
            last_measurement_update_slot = getattr(ue_measurement, "last_measurement_update_slot", None)
            if last_measurement_update_slot is not None:
                return max(0, int(getattr(state, "slot_id", 0)) - int(last_measurement_update_slot))
            return int(getattr(ue_measurement, "age_slots", 0) or 0)
        measurement_state = getattr(state, "measurement", None)
        if measurement_state is not None:
            last_measurement_update_slot = getattr(measurement_state, "last_measurement_update_slot", None)
            if last_measurement_update_slot is not None:
                return max(0, int(getattr(state, "slot_id", 0)) - int(last_measurement_update_slot))
        connection = self._get_connection(state, ue_id)
        if connection is not None:
            last_ack_slot = getattr(connection, "last_ack_slot", None)
            if last_ack_slot is not None:
                return max(0, int(getattr(state, "slot_id", 0)) - int(last_ack_slot))
        return 0

    def _estimate_channel_quality_confidence(
        self,
        *,
        cqi_validity_age_slots: int,
        measurement_age_slots: int,
        cqi_value: int,
    ) -> float:
        cqi_component = max(0.0, min(1.0, float(cqi_value) / 15.0))
        staleness_penalty = min(
            0.85,
            0.06 * float(max(0, int(cqi_validity_age_slots)))
            + 0.04 * float(max(0, int(measurement_age_slots))),
        )
        return max(0.1, min(1.0, 0.35 + 0.65 * cqi_component - staleness_penalty))

    def _allocate_observed_bytes_by_share(
        self,
        *,
        total_observed_bytes: int,
        true_bytes_by_key: dict[str, int],
    ) -> dict[str, int]:
        normalized_total_observed_bytes = max(0, int(total_observed_bytes))
        normalized_true_bytes_by_key = {
            str(key): max(0, int(value))
            for key, value in true_bytes_by_key.items()
        }
        if len(normalized_true_bytes_by_key) == 0:
            return {}
        total_true_bytes = int(sum(normalized_true_bytes_by_key.values()))
        if total_true_bytes <= 0:
            return {str(key): 0 for key in normalized_true_bytes_by_key}

        raw_allocations = {
            str(key): float(normalized_total_observed_bytes) * float(value) / float(total_true_bytes)
            for key, value in normalized_true_bytes_by_key.items()
        }
        allocations = {str(key): int(value) for key, value in raw_allocations.items()}
        remaining_bytes = normalized_total_observed_bytes - int(sum(allocations.values()))
        if remaining_bytes > 0:
            allocation_order = sorted(
                normalized_true_bytes_by_key.keys(),
                key=lambda key: (
                    raw_allocations[str(key)] - float(allocations[str(key)]),
                    normalized_true_bytes_by_key[str(key)],
                    str(key),
                ),
                reverse=True,
            )
            for key in allocation_order[:remaining_bytes]:
                allocations[str(key)] += 1
        return allocations

    def _build_logical_channel_demands(
        self,
        state: Any,
        ue: Any,
        ue_measurement: Any | None,
    ) -> list[LogicalChannelDemand]:
        logical_channel_groups = {
            int(getattr(logical_channel_group, "lcg_id", 0)): logical_channel_group
            for logical_channel_group in self._get_logical_channel_groups(ue)
        }
        hol_delay_ms = self._estimate_hol_delay_ms(state, ue, ue_measurement)
        logical_channels = self._get_logical_channels(ue)
        true_bytes_by_lc_id = {
            str(getattr(logical_channel, "logical_channel_id", "")): int(getattr(logical_channel, "queue_bytes", 0))
            for logical_channel in logical_channels
            if str(getattr(logical_channel, "logical_channel_id", "")) != ""
        }
        total_true_backlog_bytes = int(sum(true_bytes_by_lc_id.values()))
        observed_total_backlog_bytes = int(
            getattr(ue_measurement, "queue_bytes", total_true_backlog_bytes)
            if ue_measurement is not None
            else total_true_backlog_bytes
        )
        observed_backlog_by_lc_id = self._allocate_observed_bytes_by_share(
            total_observed_bytes=observed_total_backlog_bytes,
            true_bytes_by_key=true_bytes_by_lc_id,
        )
        measurement_age_slots = int(getattr(ue_measurement, "age_slots", 0) or 0) if ue_measurement is not None else 0
        logical_channel_demands: list[LogicalChannelDemand] = []
        for logical_channel in logical_channels:
            logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
            if logical_channel_id == "":
                continue
            backlog_bytes = int(getattr(logical_channel, "queue_bytes", 0))
            observed_backlog_bytes = int(observed_backlog_by_lc_id.get(logical_channel_id, backlog_bytes))
            lcg_id = int(getattr(logical_channel, "lcg_id", 0))
            logical_channel_group = logical_channel_groups.get(lcg_id)
            bsr_like_bytes = backlog_bytes
            bsr_reported_bytes = backlog_bytes
            bsr_pending_bytes = 0
            bsr_age_slots = 0
            if logical_channel_group is not None:
                bsr_reported_bytes = int(
                    getattr(logical_channel_group, "bsr_reported_bytes", backlog_bytes)
                )
                bsr_pending_bytes = max(0, int(getattr(logical_channel_group, "bsr_pending_bytes", 0)))
                bsr_age_slots = int(getattr(logical_channel_group, "bsr_age_slots", 0) or 0)
                bsr_like_bytes = int(
                    max(
                        bsr_reported_bytes,
                        backlog_bytes,
                    )
                    + bsr_pending_bytes
                )
            five_qi = self._derive_five_qi(
                slice_id=str(getattr(logical_channel, "slice_id", getattr(ue, "slice_id", ""))),
                traffic_type=str(
                    getattr(logical_channel, "traffic_type", getattr(ue, "traffic_type", ""))
                ),
                priority=int(getattr(logical_channel, "priority", getattr(ue, "priority", 0))),
            )
            demand_bytes = min(
                int(backlog_bytes),
                max(int(observed_backlog_bytes), int(bsr_like_bytes)),
            )
            logical_channel_demands.append(
                LogicalChannelDemand(
                    logical_channel_id=logical_channel_id,
                    drb_id=(
                        int(getattr(logical_channel, "drb_id", 0))
                        if getattr(logical_channel, "drb_id", None) is not None
                        else None
                    ),
                    lcg_id=lcg_id,
                    true_backlog_bytes=backlog_bytes,
                    observed_backlog_bytes=int(observed_backlog_bytes),
                    demand_bytes=int(demand_bytes),
                    backlog_bytes=backlog_bytes,
                    hol_delay_ms=float(getattr(logical_channel, "hol_delay_ms", hol_delay_ms)),
                    priority=int(getattr(logical_channel, "priority", getattr(ue, "priority", 0))),
                    packet_delay_budget_ms=float(
                        getattr(
                            logical_channel,
                            "packet_delay_budget_ms",
                            getattr(ue, "latency_budget_ms", 0.0),
                        )
                    ),
                    qos_class=self._derive_qos_class(
                        slice_id=str(getattr(logical_channel, "slice_id", getattr(ue, "slice_id", ""))),
                        traffic_type=str(
                            getattr(logical_channel, "traffic_type", getattr(ue, "traffic_type", ""))
                        ),
                        priority=int(getattr(logical_channel, "priority", getattr(ue, "priority", 0))),
                    ),
                    five_qi=five_qi,
                    slice_id=str(getattr(logical_channel, "slice_id", getattr(ue, "slice_id", ""))),
                    traffic_type=str(
                        getattr(logical_channel, "traffic_type", getattr(ue, "traffic_type", ""))
                    ),
                    bsr_like_bytes=int(bsr_like_bytes),
                    bsr_reported_bytes=int(bsr_reported_bytes),
                    bsr_pending_bytes=int(bsr_pending_bytes),
                    bsr_age_slots=int(bsr_age_slots),
                    measurement_age_slots=int(measurement_age_slots),
                    pbr_credit_bytes=float(getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0),
                )
            )
        return logical_channel_demands

    def _build_logical_channel_group_demands(
        self,
        ue: Any,
        logical_channel_demands: list[LogicalChannelDemand],
    ) -> list[LogicalChannelGroupDemand]:
        logical_channel_groups = {
            int(getattr(logical_channel_group, "lcg_id", 0)): logical_channel_group
            for logical_channel_group in self._get_logical_channel_groups(ue)
        }
        demands_by_lcg_id: dict[int, list[LogicalChannelDemand]] = {}
        for logical_channel_demand in logical_channel_demands:
            demands_by_lcg_id.setdefault(int(logical_channel_demand.lcg_id or 0), []).append(
                logical_channel_demand
            )

        logical_channel_group_demands: list[LogicalChannelGroupDemand] = []
        for lcg_id, demands in sorted(demands_by_lcg_id.items(), key=lambda item: int(item[0])):
            logical_channel_group = logical_channel_groups.get(int(lcg_id))
            dominant_priority = max(int(demand.priority) for demand in demands)
            min_delay_budget_ms = min(float(demand.packet_delay_budget_ms) for demand in demands)
            qos_class = max(
                demands,
                key=lambda demand: (int(demand.priority), -float(demand.packet_delay_budget_ms), int(demand.demand_bytes)),
            ).qos_class
            five_qi_values = [demand.five_qi for demand in demands if demand.five_qi is not None]
            logical_channel_group_demands.append(
                LogicalChannelGroupDemand(
                    lcg_id=int(lcg_id),
                    logical_channel_ids=[str(demand.logical_channel_id) for demand in demands],
                    drb_ids=sorted(
                        {
                            int(demand.drb_id)
                            for demand in demands
                            if demand.drb_id is not None
                        }
                    ),
                    true_backlog_bytes=int(sum(int(demand.true_backlog_bytes) for demand in demands)),
                    observed_backlog_bytes=int(sum(int(demand.observed_backlog_bytes) for demand in demands)),
                    demand_bytes=int(sum(int(demand.demand_bytes) for demand in demands)),
                    prioritized_queue_bytes=int(
                        sum(
                            int(demand.true_backlog_bytes)
                            for demand in demands
                            if float(demand.pbr_credit_bytes) > 0.0
                        )
                    ),
                    dominant_priority=int(dominant_priority),
                    min_packet_delay_budget_ms=float(min_delay_budget_ms),
                    qos_class=str(qos_class),
                    five_qi=(min(five_qi_values) if len(five_qi_values) > 0 else None),
                    bsr_reported_bytes=int(
                        getattr(logical_channel_group, "bsr_reported_bytes", sum(int(demand.bsr_reported_bytes) for demand in demands))
                    ) if logical_channel_group is not None else int(sum(int(demand.bsr_reported_bytes) for demand in demands)),
                    bsr_pending_bytes=int(
                        getattr(logical_channel_group, "bsr_pending_bytes", sum(int(demand.bsr_pending_bytes) for demand in demands))
                    ) if logical_channel_group is not None else int(sum(int(demand.bsr_pending_bytes) for demand in demands)),
                    bsr_age_slots=int(
                        getattr(logical_channel_group, "bsr_age_slots", 0) or 0
                    ) if logical_channel_group is not None else int(
                        max((int(demand.bsr_age_slots) for demand in demands), default=0)
                    ),
                    bsr_triggered=bool(
                        getattr(logical_channel_group, "bsr_triggered", False)
                    ) if logical_channel_group is not None else bool(
                        any(int(demand.bsr_pending_bytes) > 0 for demand in demands)
                    ),
                    pbr_credit_bytes=float(
                        getattr(logical_channel_group, "pbr_credit_bytes", 0.0) or 0.0
                    ) if logical_channel_group is not None else float(
                        sum(float(demand.pbr_credit_bytes) for demand in demands)
                    ),
                )
            )
        return logical_channel_group_demands

    def _build_ue_scheduling_demand(self, state: Any, ue: Any) -> UeSchedulingDemand:
        ue_id = str(getattr(ue, "ue_id", ""))
        ue_measurement = self._get_measurement_for_ue(state, ue_id)
        link = self._get_link_state(state, ue_id)
        logical_channel_demands = self._build_logical_channel_demands(state, ue, ue_measurement)
        logical_channel_group_demands = self._build_logical_channel_group_demands(
            ue,
            logical_channel_demands,
        )
        logical_channel_groups = self._get_logical_channel_groups(ue)

        backlog_bytes = int(getattr(ue, "queue_bytes", 0))
        if len(logical_channel_demands) > 0:
            backlog_bytes = int(
                sum(int(getattr(logical_channel_demand, "true_backlog_bytes", 0)) for logical_channel_demand in logical_channel_demands)
            )

        observed_backlog_bytes = backlog_bytes
        measurement_age_slots = 0
        if ue_measurement is not None:
            observed_backlog_bytes = int(getattr(ue_measurement, "queue_bytes", backlog_bytes))
            measurement_age_slots = int(getattr(ue_measurement, "age_slots", 0) or 0)

        dominant_priority = int(getattr(ue, "priority", 0))
        if len(logical_channel_demands) > 0:
            dominant_priority = max(
                int(getattr(logical_channel_demand, "priority", dominant_priority))
                for logical_channel_demand in logical_channel_demands
            )

        bsr_like_bytes = backlog_bytes
        if len(logical_channel_groups) > 0:
            bsr_like_bytes = int(
                sum(
                    max(
                        int(getattr(logical_channel_group, "bsr_reported_bytes", getattr(logical_channel_group, "queue_bytes", 0))),
                        int(getattr(logical_channel_group, "queue_bytes", 0)),
                    )
                    + max(0, int(getattr(logical_channel_group, "bsr_pending_bytes", 0)))
                    for logical_channel_group in logical_channel_groups
                )
            )

        cqi_validity_age_slots = self._estimate_csi_validity_age_slots(state, ue_id, ue_measurement)
        qos_class = self._derive_qos_class(
            slice_id=str(getattr(ue, "slice_id", "")),
            traffic_type=str(getattr(ue, "traffic_type", "")),
            priority=dominant_priority,
        )
        five_qi = self._derive_five_qi(
            slice_id=str(getattr(ue, "slice_id", "")),
            traffic_type=str(getattr(ue, "traffic_type", "")),
            priority=dominant_priority,
        )
        current_slot = self._get_current_slot_id(state)
        active_harq_process = self._get_active_harq_process(ue, slot_id=current_slot)
        harq_pending_bytes = self._get_pending_retx_bytes(ue, slot_id=current_slot)
        retransmission_round = int(
            getattr(
                active_harq_process,
                "retransmission_round",
                getattr(getattr(ue, "harq", None), "retransmission_count", 0),
            )
        )
        retx_deadline_slot = getattr(active_harq_process, "retx_deadline_slot", None)
        timeout_slot = getattr(active_harq_process, "timeout_slot", None)
        harq_deadline_pressure = 0.0
        if retx_deadline_slot is not None:
            harq_deadline_pressure = max(
                0.0,
                float(4 - max(0, int(retx_deadline_slot) - int(current_slot))),
            ) * 50.0
        if timeout_slot is not None:
            harq_deadline_pressure = max(
                float(harq_deadline_pressure),
                max(0.0, float(3 - max(0, int(timeout_slot) - int(current_slot)))) * 75.0,
            )
        bsr_reported_bytes = 0
        bsr_pending_bytes = 0
        if len(logical_channel_group_demands) > 0:
            bsr_reported_bytes = int(
                sum(int(getattr(logical_channel_group_demand, "bsr_reported_bytes", 0)) for logical_channel_group_demand in logical_channel_group_demands)
            )
            bsr_pending_bytes = int(
                sum(int(getattr(logical_channel_group_demand, "bsr_pending_bytes", 0)) for logical_channel_group_demand in logical_channel_group_demands)
            )
        demand_bytes = min(
            int(backlog_bytes),
            max(int(observed_backlog_bytes), int(bsr_like_bytes)),
        )
        cqi_value = int(getattr(link, "cqi", getattr(ue, "cqi", 0)) if link is not None else getattr(ue, "cqi", 0))
        channel_quality_confidence = self._estimate_channel_quality_confidence(
            cqi_validity_age_slots=int(cqi_validity_age_slots),
            measurement_age_slots=int(measurement_age_slots),
            cqi_value=int(cqi_value),
        )
        demand_reason = "bsr+measurement"
        if int(harq_pending_bytes) > 0:
            demand_reason = "harq_pending"
        elif int(bsr_pending_bytes) > 0:
            demand_reason = "bsr_pending"
        elif int(measurement_age_slots) > 0:
            demand_reason = "delayed_measurement"
        if float(channel_quality_confidence) < 0.5:
            demand_reason = f"{demand_reason}+low_confidence_channel"

        return UeSchedulingDemand(
            ue_id=ue_id,
            true_backlog_bytes=int(backlog_bytes),
            demand_bytes=int(demand_bytes),
            backlog_bytes=int(backlog_bytes),
            observed_backlog_bytes=int(observed_backlog_bytes),
            hol_delay_ms=float(self._estimate_hol_delay_ms(state, ue, ue_measurement)),
            priority=int(dominant_priority),
            qos_class=qos_class,
            five_qi=five_qi,
            slice_id=str(getattr(ue, "slice_id", "")),
            traffic_type=str(getattr(ue, "traffic_type", "")),
            bsr_like_bytes=int(bsr_like_bytes),
            bsr_reported_bytes=int(bsr_reported_bytes),
            bsr_pending_bytes=int(bsr_pending_bytes),
            harq_pending_bytes=int(harq_pending_bytes),
            harq_urgency_score=float(harq_pending_bytes)
            + float(retransmission_round) * 100.0
            + float(harq_deadline_pressure),
            cqi=int(cqi_value),
            ri=int(getattr(link, "ri", getattr(ue, "ri", 1)) if link is not None else getattr(ue, "ri", 1)),
            pmi=int(getattr(link, "pmi", getattr(ue, "pmi", 0)) if link is not None else getattr(ue, "pmi", 0)),
            cqi_validity_age_slots=int(cqi_validity_age_slots),
            ri_validity_age_slots=int(cqi_validity_age_slots),
            pmi_validity_age_slots=int(cqi_validity_age_slots),
            measurement_age_slots=int(measurement_age_slots),
            channel_quality_confidence=float(channel_quality_confidence),
            connection_state=str(
                getattr(self._get_connection(state, ue_id), "connection_state", "connected")
            ),
            demand_source="mac_state+measurement_metadata",
            demand_reason=str(demand_reason),
            logical_channel_demands=logical_channel_demands,
            logical_channel_group_demands=logical_channel_group_demands,
        )

    def _build_scheduling_demands(
        self,
        state: Any,
    ) -> tuple[list[UeSchedulingDemand], dict[str, Any], SchedulingStageTrace]:
        ue_map = {str(getattr(ue, "ue_id", "")): ue for ue in getattr(state, "ues", [])}
        feasible_demands: list[UeSchedulingDemand] = []
        filtered_out_reasons: dict[str, str] = {}

        for ue in getattr(state, "ues", []):
            ue_id = str(getattr(ue, "ue_id", ""))
            if ue_id == "":
                continue
            if not self._is_ue_connected(state, ue):
                filtered_out_reasons[ue_id] = "not_connected"
                continue
            self._prune_expired_harq_processes(state, ue)

            demand = self._build_ue_scheduling_demand(state, ue)
            if int(demand.requested_bytes) <= 0:
                filtered_out_reasons[ue_id] = "no_demand"
                continue
            feasible_demands.append(demand)

        stage_trace = SchedulingStageTrace(
            demand_source="mac_state+measurement_metadata",
            demand_visible_ue_ids=[str(demand.ue_id) for demand in feasible_demands if int(getattr(demand, "demand_bytes", 0)) > 0],
            low_confidence_channel_ue_ids=[
                str(demand.ue_id)
                for demand in feasible_demands
                if float(getattr(demand, "channel_quality_confidence", 1.0)) < 0.5
            ],
            stale_csi_ue_ids=[
                str(demand.ue_id)
                for demand in feasible_demands
                if max(
                    int(getattr(demand, "cqi_validity_age_slots", 0)),
                    int(getattr(demand, "ri_validity_age_slots", 0)),
                    int(getattr(demand, "pmi_validity_age_slots", 0)),
                ) >= 4
            ],
            harq_urgent_ue_ids=[
                str(demand.ue_id)
                for demand in feasible_demands
                if float(getattr(demand, "harq_urgency_score", 0.0)) > 0.0
            ],
            feasible_candidate_ue_ids=[str(demand.ue_id) for demand in feasible_demands],
            filtered_out_ue_ids=sorted(filtered_out_reasons.keys()),
            filtered_out_reasons=dict(filtered_out_reasons),
            rejection_reasons_by_ue=dict(filtered_out_reasons),
        )
        return feasible_demands, ue_map, stage_trace

    def _build_constraint_hits(
        self,
        state: Any,
        ue: Any,
        demand: UeSchedulingDemand,
        *,
        is_retx: bool = False,
    ) -> list[str]:
        constraint_hits: list[str] = []
        if float(getattr(demand, "channel_quality_confidence", 1.0)) < 0.5:
            constraint_hits.append("low_channel_confidence")
        if max(
            int(getattr(demand, "cqi_validity_age_slots", 0)),
            int(getattr(demand, "ri_validity_age_slots", 0)),
            int(getattr(demand, "pmi_validity_age_slots", 0)),
        ) >= 4:
            constraint_hits.append("stale_csi")
        layer_count = int(self._get_layer_count(state, ue))
        if int(getattr(demand, "ri", layer_count) or layer_count) > int(layer_count):
            constraint_hits.append("rank_limited")
        if is_retx:
            constraint_hits.append("harq_retx")
        elif self._is_poor_link_probe_candidate(ue, demand, is_retx=False):
            constraint_hits.append("poor_link_probe")
        return constraint_hits

    def _build_selection_reason(
        self,
        *,
        scheduler_mode: str,
        demand: UeSchedulingDemand,
        is_retx: bool,
    ) -> str:
        if is_retx:
            return "selected_for_harq_retransmission"
        if scheduler_mode == "latency_aware":
            return "selected_for_latency_priority"
        if scheduler_mode == "slice_aware":
            return "selected_for_slice_priority"
        if scheduler_mode == "round_robin":
            return "selected_for_round_robin_turn"
        return "selected_for_scheduler_policy"

    def _build_rejection_reason(
        self,
        *,
        scheduler_mode: str,
        demand: UeSchedulingDemand,
        is_retx: bool,
        selected_retx_count: int,
        remaining_new_tx_slots: int,
    ) -> str:
        if is_retx:
            return "retx_not_selected_due_to_lower_harq_urgency"
        if selected_retx_count > 0 and remaining_new_tx_slots <= 0:
            return "new_tx_preempted_by_retransmissions"
        if scheduler_mode == "latency_aware":
            return "new_tx_not_selected_under_latency_priority"
        if scheduler_mode == "slice_aware":
            return "new_tx_not_selected_under_slice_priority"
        if scheduler_mode == "round_robin":
            return "new_tx_not_selected_in_current_round"
        return "new_tx_not_selected_under_scheduler_policy"

    def _build_resource_pressure_source(
        self,
        *,
        is_retx: bool,
        total_candidate_count: int,
        selected_retx_count: int,
    ) -> str:
        if is_retx and total_candidate_count > 1:
            return "shared_slot_budget_under_harq_pressure"
        if not is_retx and selected_retx_count > 0:
            return "remaining_budget_after_harq_reservation"
        if total_candidate_count > 1:
            return "shared_slot_budget"
        return "grant_local_fit"

    def _summarize_logical_channel_allocations(self, grant: UeGrant) -> dict[str, object]:
        logical_channel_allocations = list(getattr(grant, "logical_channel_allocations", []) or [])
        return {
            "logical_channel_count": int(len(logical_channel_allocations)),
            "scheduled_bytes_total": int(
                sum(int(getattr(allocation, "scheduled_bytes", 0)) for allocation in logical_channel_allocations)
            ),
            "prioritized_bytes_total": int(
                sum(int(getattr(allocation, "prioritized_bytes", 0)) for allocation in logical_channel_allocations)
            ),
            "best_effort_bytes_total": int(
                sum(int(getattr(allocation, "best_effort_bytes", 0)) for allocation in logical_channel_allocations)
            ),
            "retx_logical_channel_count": int(
                sum(1 for allocation in logical_channel_allocations if bool(getattr(allocation, "is_retx", False)))
            ),
            "scheduled_lcg_count": int(
                len(
                    {
                        int(getattr(allocation, "lcg_id", 0))
                        for allocation in logical_channel_allocations
                        if getattr(allocation, "lcg_id", None) is not None
                    }
                )
            ),
        }

    def _get_logical_channels(self, ue: Any) -> list[Any]:
        logical_channels = list(getattr(ue, "logical_channels", []) or [])
        return [
            logical_channel
            for logical_channel in logical_channels
            if str(getattr(logical_channel, "logical_channel_id", "")) != ""
        ]

    def _get_logical_channel_groups(self, ue: Any) -> list[Any]:
        logical_channel_groups = list(getattr(ue, "logical_channel_groups", []) or [])
        if len(logical_channel_groups) > 0:
            return [
                logical_channel_group
                for logical_channel_group in logical_channel_groups
                if getattr(logical_channel_group, "lcg_id", None) is not None
            ]

        groups_by_lcg_id: dict[int, LogicalChannelGroupState] = {}
        for logical_channel in self._get_logical_channels(ue):
            lcg_id = int(getattr(logical_channel, "lcg_id", 0))
            logical_channel_group = groups_by_lcg_id.setdefault(
                lcg_id,
                LogicalChannelGroupState(lcg_id=lcg_id),
            )
            logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
            if logical_channel_id != "":
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
        return sorted(
            groups_by_lcg_id.values(),
            key=lambda logical_channel_group: int(logical_channel_group.lcg_id),
        )

    def _get_logical_channel_queue_bytes(self, ue: Any, logical_channel_id: str) -> int:
        for logical_channel in self._get_logical_channels(ue):
            if str(getattr(logical_channel, "logical_channel_id", "")) == str(logical_channel_id):
                return int(getattr(logical_channel, "queue_bytes", 0))
        return 0

    def _build_logical_channel_demand_map(
        self,
        ue_demand: UeSchedulingDemand | None,
    ) -> dict[str, LogicalChannelDemand]:
        if ue_demand is None:
            return {}
        return {
            str(logical_channel_demand.logical_channel_id): logical_channel_demand
            for logical_channel_demand in list(getattr(ue_demand, "logical_channel_demands", []) or [])
            if str(getattr(logical_channel_demand, "logical_channel_id", "")) != ""
        }

    def _build_logical_channel_group_demand_map(
        self,
        ue_demand: UeSchedulingDemand | None,
    ) -> dict[int, LogicalChannelGroupDemand]:
        if ue_demand is None:
            return {}
        return {
            int(logical_channel_group_demand.lcg_id): logical_channel_group_demand
            for logical_channel_group_demand in list(getattr(ue_demand, "logical_channel_group_demands", []) or [])
        }

    def _order_logical_channels_within_group(
        self,
        ue: Any,
        lcg_id: int,
        ue_demand: UeSchedulingDemand | None = None,
    ) -> list[Any]:
        logical_channel_demand_map = self._build_logical_channel_demand_map(ue_demand)
        logical_channels = [
            logical_channel
            for logical_channel in self._get_logical_channels(ue)
            if int(getattr(logical_channel, "lcg_id", 0)) == int(lcg_id)
            and int(getattr(logical_channel, "queue_bytes", 0)) > 0
        ]
        return sorted(
            logical_channels,
            key=lambda logical_channel: (
                -int(
                    getattr(
                        logical_channel_demand_map.get(
                            str(getattr(logical_channel, "logical_channel_id", ""))
                        ),
                        "priority",
                        getattr(logical_channel, "priority", getattr(ue, "priority", 0)),
                    )
                ),
                float(
                    getattr(
                        logical_channel_demand_map.get(
                            str(getattr(logical_channel, "logical_channel_id", ""))
                        ),
                        "packet_delay_budget_ms",
                        getattr(
                            logical_channel,
                            "packet_delay_budget_ms",
                            getattr(ue, "latency_budget_ms", 0.0),
                        ),
                    )
                ),
                -int(
                    getattr(
                        logical_channel_demand_map.get(
                            str(getattr(logical_channel, "logical_channel_id", ""))
                        ),
                        "demand_bytes",
                        getattr(logical_channel, "queue_bytes", 0),
                    )
                ),
                str(getattr(logical_channel, "logical_channel_id", "")),
            ),
        )

    def _get_lcg_headline(
        self,
        ue: Any,
        logical_channel_group: Any,
        ue_demand: UeSchedulingDemand | None = None,
    ) -> tuple[int, float]:
        lcg_id = int(getattr(logical_channel_group, "lcg_id", 0))
        logical_channel_group_demand = self._build_logical_channel_group_demand_map(ue_demand).get(lcg_id)
        if logical_channel_group_demand is not None:
            return (
                int(getattr(logical_channel_group_demand, "dominant_priority", getattr(ue, "priority", 0))),
                float(
                    getattr(
                        logical_channel_group_demand,
                        "min_packet_delay_budget_ms",
                        getattr(ue, "latency_budget_ms", 0.0),
                    )
                ),
            )
        logical_channels = self._order_logical_channels_within_group(ue, lcg_id, ue_demand)
        if len(logical_channels) == 0:
            return int(getattr(ue, "priority", 0)), float(getattr(ue, "latency_budget_ms", 0.0))
        dominant_priority = max(
            int(getattr(logical_channel, "priority", getattr(ue, "priority", 0)))
            for logical_channel in logical_channels
        )
        min_delay_budget_ms = min(
            float(
                getattr(
                    logical_channel,
                    "packet_delay_budget_ms",
                    getattr(ue, "latency_budget_ms", 0.0),
                )
            )
            for logical_channel in logical_channels
        )
        return dominant_priority, min_delay_budget_ms

    def _get_lcg_priority_weight(
        self,
        ue: Any,
        logical_channel_group: Any,
        ue_demand: UeSchedulingDemand | None = None,
    ) -> float:
        lcg_id = int(getattr(logical_channel_group, "lcg_id", 0))
        logical_channel_group_demand = self._build_logical_channel_group_demand_map(ue_demand).get(lcg_id)
        dominant_priority, min_delay_budget_ms = self._get_lcg_headline(
            ue,
            logical_channel_group,
            ue_demand,
        )
        queue_bytes = int(
            getattr(
                logical_channel_group_demand,
                "demand_bytes",
                getattr(logical_channel_group, "queue_bytes", 0),
            )
        )
        prioritized_queue_bytes = int(
            getattr(
                logical_channel_group_demand,
                "prioritized_queue_bytes",
                getattr(logical_channel_group, "prioritized_queue_bytes", 0),
            )
        )
        bsr_reported_bytes = int(
            getattr(
                logical_channel_group_demand,
                "bsr_reported_bytes",
                getattr(logical_channel_group, "bsr_reported_bytes", queue_bytes),
            )
        )
        bsr_pending_bytes = int(
            getattr(
                logical_channel_group_demand,
                "bsr_pending_bytes",
                getattr(logical_channel_group, "bsr_pending_bytes", 0),
            )
        )
        bsr_age_slots = int(
            getattr(
                logical_channel_group_demand,
                "bsr_age_slots",
                getattr(logical_channel_group, "bsr_age_slots", 0),
            )
            or 0
        )
        bsr_trigger_bonus = 50.0 if bool(
            getattr(
                logical_channel_group_demand,
                "bsr_triggered",
                getattr(logical_channel_group, "bsr_triggered", False),
            )
        ) else 0.0
        pbr_credit_bytes = float(
            getattr(
                logical_channel_group_demand,
                "pbr_credit_bytes",
                getattr(logical_channel_group, "pbr_credit_bytes", 0.0),
            )
            or 0.0
        )
        delay_bonus = max(0.0, 100.0 - min(100.0, float(min_delay_budget_ms)))
        return (
            1.0
            + float(dominant_priority) * 100.0
            + delay_bonus
            + bsr_trigger_bonus
            + float(bsr_pending_bytes) / 10.0
            + float(bsr_reported_bytes) / 20.0
            + float(prioritized_queue_bytes) / 20.0
            + float(queue_bytes) / 50.0
            + pbr_credit_bytes / 100.0
            + float(bsr_age_slots)
        )

    def _order_logical_channel_groups_for_new_tx(
        self,
        ue: Any,
        ue_demand: UeSchedulingDemand | None = None,
    ) -> list[Any]:
        logical_channel_group_demand_map = self._build_logical_channel_group_demand_map(ue_demand)
        logical_channel_groups = [
            logical_channel_group
            for logical_channel_group in self._get_logical_channel_groups(ue)
            if int(getattr(logical_channel_group, "queue_bytes", 0)) > 0
        ]
        return sorted(
            logical_channel_groups,
            key=lambda logical_channel_group: (
                -int(self._get_lcg_headline(ue, logical_channel_group, ue_demand)[0]),
                float(self._get_lcg_headline(ue, logical_channel_group, ue_demand)[1]),
                -int(
                    bool(
                        getattr(
                            logical_channel_group_demand_map.get(
                                int(getattr(logical_channel_group, "lcg_id", 0))
                            ),
                            "bsr_triggered",
                            getattr(logical_channel_group, "bsr_triggered", False),
                        )
                    )
                ),
                -int(
                    getattr(
                        logical_channel_group_demand_map.get(
                            int(getattr(logical_channel_group, "lcg_id", 0))
                        ),
                        "bsr_pending_bytes",
                        getattr(logical_channel_group, "bsr_pending_bytes", 0),
                    )
                ),
                -int(
                    getattr(
                        logical_channel_group_demand_map.get(
                            int(getattr(logical_channel_group, "lcg_id", 0))
                        ),
                        "bsr_reported_bytes",
                        getattr(
                            logical_channel_group,
                            "bsr_reported_bytes",
                            getattr(logical_channel_group, "queue_bytes", 0),
                        ),
                    )
                ),
                -int(
                    getattr(
                        logical_channel_group_demand_map.get(
                            int(getattr(logical_channel_group, "lcg_id", 0))
                        ),
                        "prioritized_queue_bytes",
                        getattr(logical_channel_group, "prioritized_queue_bytes", 0),
                    )
                ),
                -int(
                    getattr(
                        logical_channel_group_demand_map.get(
                            int(getattr(logical_channel_group, "lcg_id", 0))
                        ),
                        "demand_bytes",
                        getattr(logical_channel_group, "queue_bytes", 0),
                    )
                ),
                int(getattr(logical_channel_group, "lcg_id", 0)),
            ),
        )

    def _order_logical_channels_for_new_tx(
        self,
        ue: Any,
        ue_demand: UeSchedulingDemand | None = None,
    ) -> list[Any]:
        ordered_logical_channels: list[Any] = []
        covered_logical_channel_ids: set[str] = set()
        for logical_channel_group in self._order_logical_channel_groups_for_new_tx(ue, ue_demand):
            for logical_channel in self._order_logical_channels_within_group(
                ue,
                int(getattr(logical_channel_group, "lcg_id", 0)),
                ue_demand,
            ):
                logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
                if logical_channel_id == "" or logical_channel_id in covered_logical_channel_ids:
                    continue
                ordered_logical_channels.append(logical_channel)
                covered_logical_channel_ids.add(logical_channel_id)

        for logical_channel in self._get_logical_channels(ue):
            logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
            if (
                logical_channel_id == ""
                or logical_channel_id in covered_logical_channel_ids
                or int(getattr(logical_channel, "queue_bytes", 0)) <= 0
            ):
                continue
            ordered_logical_channels.append(logical_channel)
        return ordered_logical_channels

    def _allocate_weighted_bytes(self, weights: list[float], total_bytes: int) -> list[int]:
        normalized_total_bytes = max(0, int(total_bytes))
        if normalized_total_bytes <= 0 or len(weights) == 0:
            return [0 for _ in weights]

        normalized_weights = [max(0.0, float(weight)) for weight in weights]
        if sum(normalized_weights) <= 0.0:
            normalized_weights = [1.0 for _ in weights]

        raw_allocations = [
            float(normalized_total_bytes) * float(weight) / float(sum(normalized_weights))
            for weight in normalized_weights
        ]
        allocations = [int(allocation) for allocation in raw_allocations]
        remaining_bytes = normalized_total_bytes - int(sum(allocations))
        if remaining_bytes > 0:
            allocation_order = sorted(
                range(len(raw_allocations)),
                key=lambda index: (
                    raw_allocations[index] - float(allocations[index]),
                    normalized_weights[index],
                    -index,
                ),
                reverse=True,
            )
            for index in allocation_order[:remaining_bytes]:
                allocations[index] += 1
        return allocations

    def _allocate_group_byte_budgets(
        self,
        *,
        group_weights: list[float],
        group_demands: list[int],
        total_bytes: int,
    ) -> list[int]:
        budgets = [0 for _ in group_demands]
        remaining_bytes = max(0, int(total_bytes))
        active_indices = [
            index for index, demand_bytes in enumerate(group_demands)
            if int(demand_bytes) > 0
        ]

        while remaining_bytes > 0 and len(active_indices) > 0:
            active_allocations = self._allocate_weighted_bytes(
                [group_weights[index] for index in active_indices],
                remaining_bytes,
            )
            assigned_bytes = 0
            for group_index, candidate_budget in zip(active_indices, active_allocations):
                remaining_demand_bytes = max(0, int(group_demands[group_index]) - int(budgets[group_index]))
                assigned_budget = min(int(candidate_budget), remaining_demand_bytes)
                budgets[group_index] += int(assigned_budget)
                assigned_bytes += int(assigned_budget)

            if assigned_bytes <= 0:
                for group_index in active_indices:
                    remaining_demand_bytes = max(0, int(group_demands[group_index]) - int(budgets[group_index]))
                    if remaining_demand_bytes <= 0 or remaining_bytes <= 0:
                        continue
                    budgets[group_index] += 1
                    remaining_bytes -= 1
                    assigned_bytes += 1
                    if remaining_bytes <= 0:
                        break
                if assigned_bytes <= 0:
                    break
            else:
                remaining_bytes -= int(assigned_bytes)

            active_indices = [
                index
                for index in active_indices
                if int(budgets[index]) < int(group_demands[index])
            ]

        return budgets

    def _allocate_bytes_across_logical_channels(
        self,
        ue: Any,
        *,
        logical_channels: list[Any],
        ue_demand: UeSchedulingDemand | None,
        grant_bytes_budget: int,
        allocations_by_logical_channel_id: dict[str, LogicalChannelGrant],
    ) -> int:
        logical_channel_demand_map = self._build_logical_channel_demand_map(ue_demand)
        remaining_bytes = max(0, int(grant_bytes_budget))

        for logical_channel in logical_channels:
            if remaining_bytes <= 0:
                break
            logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
            logical_channel_demand = logical_channel_demand_map.get(logical_channel_id)
            queue_bytes = int(getattr(logical_channel, "queue_bytes", 0))
            demand_bytes = int(
                getattr(logical_channel_demand, "demand_bytes", queue_bytes)
                if logical_channel_demand is not None
                else queue_bytes
            )
            effective_queue_bytes = min(queue_bytes, max(0, demand_bytes))
            available_credit_bytes = int(
                min(
                    float(effective_queue_bytes),
                    float(max(0.0, getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0)),
                )
            )
            prioritized_bytes = min(remaining_bytes, available_credit_bytes)
            if prioritized_bytes <= 0:
                continue
            existing_allocation = allocations_by_logical_channel_id.get(logical_channel_id)
            if existing_allocation is None:
                allocations_by_logical_channel_id[logical_channel_id] = LogicalChannelGrant(
                    logical_channel_id=logical_channel_id,
                    drb_id=int(getattr(logical_channel, "drb_id", 1)),
                    lcg_id=int(getattr(logical_channel, "lcg_id", 0)),
                    scheduled_bytes=int(prioritized_bytes),
                    queue_bytes_before=queue_bytes,
                    priority=int(getattr(logical_channel, "priority", getattr(ue, "priority", 0))),
                    packet_delay_budget_ms=float(
                        getattr(
                            logical_channel,
                            "packet_delay_budget_ms",
                            getattr(ue, "latency_budget_ms", 0.0),
                        )
                    ),
                    slice_id=str(getattr(logical_channel, "slice_id", getattr(ue, "slice_id", ""))),
                    traffic_type=str(
                        getattr(logical_channel, "traffic_type", getattr(ue, "traffic_type", ""))
                    ),
                    is_retx=False,
                    prioritized_bytes=int(prioritized_bytes),
                    best_effort_bytes=0,
                    used_pbr_credit_bytes=int(prioritized_bytes),
                    allocation_reason="pbr_credit",
                )
            else:
                existing_allocation.scheduled_bytes += int(prioritized_bytes)
                existing_allocation.prioritized_bytes += int(prioritized_bytes)
                existing_allocation.used_pbr_credit_bytes += int(prioritized_bytes)
            remaining_bytes -= int(prioritized_bytes)
            try:
                logical_channel.pbr_credit_bytes = max(
                    0.0,
                    float(getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0)
                    - float(prioritized_bytes),
                )
            except Exception:
                pass

        for logical_channel in logical_channels:
            if remaining_bytes <= 0:
                break
            logical_channel_id = str(getattr(logical_channel, "logical_channel_id", ""))
            logical_channel_demand = logical_channel_demand_map.get(logical_channel_id)
            queue_bytes = int(getattr(logical_channel, "queue_bytes", 0))
            demand_bytes = int(
                getattr(logical_channel_demand, "demand_bytes", queue_bytes)
                if logical_channel_demand is not None
                else queue_bytes
            )
            effective_queue_bytes = min(queue_bytes, max(0, demand_bytes))
            existing_allocation = allocations_by_logical_channel_id.get(logical_channel_id)
            already_allocated_bytes = (
                int(existing_allocation.scheduled_bytes) if existing_allocation is not None else 0
            )
            remaining_queue_bytes = max(0, effective_queue_bytes - already_allocated_bytes)
            best_effort_bytes = min(remaining_bytes, remaining_queue_bytes)
            if best_effort_bytes <= 0:
                continue
            if existing_allocation is None:
                allocations_by_logical_channel_id[logical_channel_id] = LogicalChannelGrant(
                    logical_channel_id=logical_channel_id,
                    drb_id=int(getattr(logical_channel, "drb_id", 1)),
                    lcg_id=int(getattr(logical_channel, "lcg_id", 0)),
                    scheduled_bytes=int(best_effort_bytes),
                    queue_bytes_before=queue_bytes,
                    priority=int(getattr(logical_channel, "priority", getattr(ue, "priority", 0))),
                    packet_delay_budget_ms=float(
                        getattr(
                            logical_channel,
                            "packet_delay_budget_ms",
                            getattr(ue, "latency_budget_ms", 0.0),
                        )
                    ),
                    slice_id=str(getattr(logical_channel, "slice_id", getattr(ue, "slice_id", ""))),
                    traffic_type=str(
                        getattr(logical_channel, "traffic_type", getattr(ue, "traffic_type", ""))
                    ),
                    is_retx=False,
                    prioritized_bytes=0,
                    best_effort_bytes=int(best_effort_bytes),
                    used_pbr_credit_bytes=0,
                    allocation_reason="best_effort",
                )
            else:
                existing_allocation.scheduled_bytes += int(best_effort_bytes)
                existing_allocation.best_effort_bytes += int(best_effort_bytes)
            remaining_bytes -= int(best_effort_bytes)

        return remaining_bytes

    def _estimate_logical_channel_credit_increment_bytes(
        self,
        state: Any,
        logical_channel: Any,
    ) -> float:
        prioritized_bit_rate_bps = getattr(logical_channel, "prioritized_bit_rate_bps", None)
        if prioritized_bit_rate_bps is None:
            return 0.0
        prioritized_bit_rate_bps_value = float(prioritized_bit_rate_bps)
        if prioritized_bit_rate_bps_value <= 0.0:
            return 0.0
        slot_duration_ms = max(float(getattr(state, "slot_duration_ms", 1.0)), 1e-6)
        return float(prioritized_bit_rate_bps_value * (slot_duration_ms / 1000.0) / 8.0)

    def _estimate_logical_channel_credit_cap_bytes(self, logical_channel: Any) -> float | None:
        prioritized_bit_rate_bps = getattr(logical_channel, "prioritized_bit_rate_bps", None)
        if prioritized_bit_rate_bps is None:
            return None
        prioritized_bit_rate_bps_value = float(prioritized_bit_rate_bps)
        if prioritized_bit_rate_bps_value <= 0.0:
            return None
        bucket_size_duration_ms = max(
            0.0,
            float(getattr(logical_channel, "bucket_size_duration_ms", 10.0)),
        )
        return float(prioritized_bit_rate_bps_value * (bucket_size_duration_ms / 1000.0) / 8.0)

    def _top_up_logical_channel_pbr_credit(self, state: Any, logical_channel: Any) -> None:
        credit_increment_bytes = self._estimate_logical_channel_credit_increment_bytes(state, logical_channel)
        current_slot_id = int(getattr(state, "slot_id", 0))
        last_credit_update_slot = getattr(logical_channel, "last_credit_update_slot", None)
        if last_credit_update_slot is None:
            elapsed_slots = 1
        elif current_slot_id <= int(last_credit_update_slot):
            elapsed_slots = 0
        else:
            elapsed_slots = current_slot_id - int(last_credit_update_slot)

        current_credit_bytes = float(getattr(logical_channel, "pbr_credit_bytes", 0.0) or 0.0)
        updated_credit_bytes = current_credit_bytes + float(elapsed_slots) * float(credit_increment_bytes)
        credit_cap_bytes = self._estimate_logical_channel_credit_cap_bytes(logical_channel)
        if credit_cap_bytes is not None:
            updated_credit_bytes = min(float(credit_cap_bytes), float(updated_credit_bytes))
        try:
            logical_channel.pbr_credit_bytes = max(0.0, float(updated_credit_bytes))
            logical_channel.last_credit_update_slot = int(current_slot_id)
        except Exception:
            pass

    def _refresh_logical_channel_pbr_credits(self, state: Any) -> None:
        for ue in getattr(state, "ues", []):
            if not self._is_ue_connected(state, ue):
                continue
            for logical_channel in self._get_logical_channels(ue):
                self._top_up_logical_channel_pbr_credit(state, logical_channel)

    def _build_new_tx_logical_channel_allocations(
        self,
        ue: Any,
        *,
        ue_demand: UeSchedulingDemand | None,
        grant_bytes_budget: int,
    ) -> list[LogicalChannelGrant]:
        logical_channel_group_demand_map = self._build_logical_channel_group_demand_map(ue_demand)
        ordered_logical_channel_groups = self._order_logical_channel_groups_for_new_tx(ue, ue_demand)
        allocations_by_logical_channel_id: dict[str, LogicalChannelGrant] = {}
        if len(ordered_logical_channel_groups) == 0:
            ordered_logical_channels = self._order_logical_channels_for_new_tx(ue, ue_demand)
            self._allocate_bytes_across_logical_channels(
                ue,
                logical_channels=ordered_logical_channels,
                ue_demand=ue_demand,
                grant_bytes_budget=int(grant_bytes_budget),
                allocations_by_logical_channel_id=allocations_by_logical_channel_id,
            )
            return list(allocations_by_logical_channel_id.values())

        group_demands = [
            min(
                int(getattr(logical_channel_group, "queue_bytes", 0)),
                max(
                    0,
                    int(
                        getattr(
                            logical_channel_group_demand_map.get(int(getattr(logical_channel_group, "lcg_id", 0))),
                            "demand_bytes",
                            getattr(logical_channel_group, "queue_bytes", 0),
                        )
                    ),
                ),
            )
            for logical_channel_group in ordered_logical_channel_groups
        ]
        group_weights = [
            float(self._get_lcg_priority_weight(ue, logical_channel_group, ue_demand))
            for logical_channel_group in ordered_logical_channel_groups
        ]
        group_budgets = self._allocate_group_byte_budgets(
            group_weights=group_weights,
            group_demands=group_demands,
            total_bytes=int(grant_bytes_budget),
        )
        remaining_bytes = max(0, int(grant_bytes_budget))
        for logical_channel_group, group_budget in zip(ordered_logical_channel_groups, group_budgets):
            if remaining_bytes <= 0 or int(group_budget) <= 0:
                continue
            group_channels = self._order_logical_channels_within_group(
                ue,
                int(getattr(logical_channel_group, "lcg_id", 0)),
                ue_demand,
            )
            unused_group_bytes = self._allocate_bytes_across_logical_channels(
                ue,
                logical_channels=group_channels,
                ue_demand=ue_demand,
                grant_bytes_budget=int(group_budget),
                allocations_by_logical_channel_id=allocations_by_logical_channel_id,
            )
            remaining_bytes -= max(0, int(group_budget) - int(unused_group_bytes))

        if remaining_bytes > 0:
            ordered_logical_channels = self._order_logical_channels_for_new_tx(ue, ue_demand)
            self._allocate_bytes_across_logical_channels(
                ue,
                logical_channels=ordered_logical_channels,
                ue_demand=ue_demand,
                grant_bytes_budget=remaining_bytes,
                allocations_by_logical_channel_id=allocations_by_logical_channel_id,
            )
        return list(allocations_by_logical_channel_id.values())

    def _build_retx_logical_channel_allocations(
        self,
        ue: Any,
        *,
        grant_bytes_budget: int,
    ) -> list[LogicalChannelGrant]:
        harq_state = getattr(ue, "harq", None)
        pending_entries = list(getattr(harq_state, "pending_retx_logical_channels", []) or [])
        if len(pending_entries) == 0:
            fallback_allocations = self._build_new_tx_logical_channel_allocations(
                ue,
                ue_demand=None,
                grant_bytes_budget=grant_bytes_budget,
            )
            for allocation in fallback_allocations:
                allocation.is_retx = True
            return fallback_allocations

        ordered_entries = sorted(
            pending_entries,
            key=lambda entry: (
                -int(getattr(entry, "priority", getattr(ue, "priority", 0))),
                float(
                    getattr(
                        entry,
                        "packet_delay_budget_ms",
                        getattr(ue, "latency_budget_ms", 0.0),
                    )
                ),
                -int(getattr(entry, "pending_bytes", 0)),
                str(getattr(entry, "logical_channel_id", "")),
            ),
        )
        remaining_bytes = max(0, int(grant_bytes_budget))
        allocations: list[LogicalChannelGrant] = []
        for entry in ordered_entries:
            if remaining_bytes <= 0:
                break
            pending_bytes = int(getattr(entry, "pending_bytes", 0))
            scheduled_bytes = min(remaining_bytes, pending_bytes)
            if scheduled_bytes <= 0:
                continue
            logical_channel_id = str(getattr(entry, "logical_channel_id", ""))
            allocations.append(
                LogicalChannelGrant(
                    logical_channel_id=logical_channel_id,
                    drb_id=(
                        int(getattr(entry, "drb_id", 1))
                        if getattr(entry, "drb_id", None) is not None
                        else None
                    ),
                    lcg_id=(
                        int(getattr(entry, "lcg_id", 0))
                        if getattr(entry, "lcg_id", None) is not None
                        else None
                    ),
                    scheduled_bytes=int(scheduled_bytes),
                    queue_bytes_before=self._get_logical_channel_queue_bytes(ue, logical_channel_id),
                    priority=int(getattr(entry, "priority", getattr(ue, "priority", 0))),
                    packet_delay_budget_ms=float(
                        getattr(
                            entry,
                            "packet_delay_budget_ms",
                            getattr(ue, "latency_budget_ms", 0.0),
                        )
                    ),
                    slice_id=str(getattr(entry, "slice_id", getattr(ue, "slice_id", ""))),
                    traffic_type=str(getattr(entry, "traffic_type", getattr(ue, "traffic_type", ""))),
                    is_retx=True,
                    allocation_reason="harq_retransmission",
                )
            )
            remaining_bytes -= int(scheduled_bytes)
        return allocations

    def _assign_logical_channel_allocations(
        self,
        grants: list[UeGrant],
        ue_map: dict[str, Any],
        demand_map: dict[str, UeSchedulingDemand] | None = None,
    ) -> None:
        normalized_demand_map = demand_map or {}
        for grant in grants:
            ue = ue_map.get(str(grant.ue_id))
            if ue is None or not grant.is_active:
                grant.logical_channel_allocations = []
                continue
            if bool(getattr(grant, "is_retx", False)):
                grant.logical_channel_allocations = self._build_retx_logical_channel_allocations(
                    ue,
                    grant_bytes_budget=int(getattr(grant, "grant_bytes_budget", 0)),
                )
            else:
                grant.logical_channel_allocations = self._build_new_tx_logical_channel_allocations(
                    ue,
                    ue_demand=normalized_demand_map.get(str(grant.ue_id)),
                    grant_bytes_budget=int(getattr(grant, "grant_bytes_budget", 0)),
                )

    def _get_beamforming_state(self, state: Any, ue_id: str) -> Any | None:
        beamforming_map = getattr(state, "beamforming", {})
        if isinstance(beamforming_map, dict):
            return beamforming_map.get(str(ue_id))
        return None

    def _get_layer_count(self, state: Any, ue: Any) -> int:
        beamforming_state = self._get_beamforming_state(state, str(getattr(ue, "ue_id", "")))
        if beamforming_state is not None:
            return max(1, int(getattr(beamforming_state, "layer_count", getattr(ue, "ri", 1))))
        return max(1, int(getattr(ue, "ri", getattr(ue, "rank", 1))))

    def _get_precoder_id(self, state: Any, ue: Any) -> int:
        beamforming_state = self._get_beamforming_state(state, str(getattr(ue, "ue_id", "")))
        if beamforming_state is not None:
            return int(getattr(beamforming_state, "precoder_id", getattr(getattr(state, "cell", None), "current_codebook_id", 0)))
        return int(getattr(getattr(state, "cell", None), "current_codebook_id", 0))

    def _get_beam_direction_deg(self, state: Any, ue: Any) -> float:
        beamforming_state = self._get_beamforming_state(state, str(getattr(ue, "ue_id", "")))
        if beamforming_state is not None:
            return float(getattr(beamforming_state, "beam_direction_deg", 0.0))
        return float(int(getattr(ue, "serving_beam_id", 0)) * 30.0)

    def _compute_beam_separation_deg(self, left_direction_deg: float, right_direction_deg: float) -> float:
        raw_delta = abs(float(left_direction_deg) - float(right_direction_deg)) % 360.0
        return min(raw_delta, 360.0 - raw_delta)

    def _compute_spatial_compatibility_score(self, state: Any, left_ue: Any, right_ue: Any) -> float:
        beam_separation_deg = self._compute_beam_separation_deg(
            self._get_beam_direction_deg(state, left_ue),
            self._get_beam_direction_deg(state, right_ue),
        )
        min_beam_separation_deg = self._get_min_mu_beam_separation_deg(state)
        if beam_separation_deg < min_beam_separation_deg:
            return 0.0

        max_total_layers = self._get_max_total_spatial_layers(state)
        total_layers = self._get_layer_count(state, left_ue) + self._get_layer_count(state, right_ue)
        if total_layers > max_total_layers:
            return 0.0

        normalized_separation = min(1.0, beam_separation_deg / 180.0)
        precoder_diversity = 1.0 if self._get_precoder_id(state, left_ue) != self._get_precoder_id(state, right_ue) else 0.85
        layer_headroom = max(0.1, 1.0 - (float(total_layers - 1) / float(max_total_layers + 1)))
        return max(0.0, min(1.0, normalized_separation * precoder_diversity * layer_headroom))

    def _get_requested_bytes(self, ue: Any, *, is_retx: bool) -> int:
        if is_retx:
            return self._get_pending_retx_bytes(ue)
        logical_channels = self._get_logical_channels(ue)
        if len(logical_channels) > 0:
            return int(sum(int(getattr(logical_channel, "queue_bytes", 0)) for logical_channel in logical_channels))
        return int(getattr(ue, "queue_bytes", 0))

    def _get_effective_link_quality_db(self, ue: Any) -> float:
        candidate_values = [
            float(getattr(ue, "post_eq_sinr_db", 0.0) or 0.0),
            float(getattr(ue, "pre_eq_sinr_db", 0.0) or 0.0),
            float(getattr(ue, "sinr_db", 0.0) or 0.0),
            float(getattr(ue, "radio_quality_db", 0.0) or 0.0),
        ]
        return max(candidate_values)

    def _is_poor_link_probe_candidate(
        self,
        ue: Any,
        demand: UeSchedulingDemand | None = None,
        *,
        is_retx: bool,
    ) -> bool:
        if is_retx:
            return False
        link_quality_db = self._get_effective_link_quality_db(ue)
        current_bler = float(getattr(ue, "bler", 0.0) or 0.0)
        channel_quality_confidence = float(
            getattr(demand, "channel_quality_confidence", 1.0) if demand is not None else 1.0
        )
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        cqi_value = int(getattr(demand, "cqi", getattr(ue, "cqi", 0)) if demand is not None else getattr(ue, "cqi", 0))
        harq_flush_count = int(getattr(getattr(ue, "harq", None), "harq_flush_count", 0) or 0)

        if link_quality_db <= -8.0:
            return True
        if current_bler >= 0.85:
            return True
        if link_quality_db <= -2.0 and current_bler >= 0.55:
            return True
        if channel_quality_confidence < 0.35:
            return True
        if channel_quality_confidence < 0.50 and csi_age_slots >= 6 and cqi_value <= 5:
            return True
        if harq_flush_count > 0 and link_quality_db <= 2.0 and current_bler >= 0.35:
            return True
        return False

    def _derive_poor_link_probe_requested_bytes_cap(
        self,
        ue: Any,
        demand: UeSchedulingDemand | None = None,
    ) -> int:
        link_quality_db = self._get_effective_link_quality_db(ue)
        current_bler = float(getattr(ue, "bler", 0.0) or 0.0)
        channel_quality_confidence = float(
            getattr(demand, "channel_quality_confidence", 1.0) if demand is not None else 1.0
        )
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        if link_quality_db <= -10.0 or current_bler >= 0.92:
            return 256
        if channel_quality_confidence < 0.35 or link_quality_db <= -5.0:
            return 384
        if current_bler >= 0.65 or csi_age_slots >= 6:
            return 512
        return 640

    def _poor_link_probe_score_scale(
        self,
        ue: Any,
        demand: UeSchedulingDemand | None = None,
    ) -> float:
        latency_budget_ms = float(getattr(ue, "latency_budget_ms", 50.0) or 50.0)
        slice_id = str(
            getattr(demand, "slice_id", getattr(ue, "slice_id", "")) if demand is not None else getattr(ue, "slice_id", "")
        ).strip().lower()
        if slice_id == "urllc" or latency_budget_ms <= 20.0:
            return 0.55
        return 0.35

    def _order_demands_round_robin(
        self,
        state: Any,
        demands: list[UeSchedulingDemand],
    ) -> list[UeSchedulingDemand]:
        if len(demands) == 0:
            return []
        ue_id_to_demand = {str(demand.ue_id): demand for demand in demands}
        demand_ues = []
        for ue_id in ue_id_to_demand.keys():
            ue = self._get_ue_by_id(state, ue_id)
            if ue is not None:
                demand_ues.append(ue)
        ordered_ues = self._order_round_robin(
            state,
            demand_ues,
        )
        ordered_demands = [
            ue_id_to_demand[str(getattr(ue, "ue_id", ""))]
            for ue in ordered_ues
            if str(getattr(ue, "ue_id", "")) in ue_id_to_demand
        ]
        covered_ue_ids = {str(demand.ue_id) for demand in ordered_demands}
        ordered_demands.extend(
            demand for demand in demands if str(demand.ue_id) not in covered_ue_ids
        )
        return ordered_demands

    def _order_demands_for_mode(
        self,
        state: Any,
        demands: list[UeSchedulingDemand],
        ue_map: dict[str, Any],
        scheduler_mode: str,
    ) -> list[UeSchedulingDemand]:
        if scheduler_mode == "round_robin":
            return self._order_demands_round_robin(state, demands)

        if scheduler_mode == "latency_aware":
            return sorted(
                demands,
                key=lambda demand: self._score_for_mode(
                    state,
                    ue_map[str(demand.ue_id)],
                    scheduler_mode,
                    demand=demand,
                ),
                reverse=True,
            )

        if scheduler_mode == "slice_aware":
            return sorted(
                demands,
                key=lambda demand: self._score_for_mode(
                    state,
                    ue_map[str(demand.ue_id)],
                    scheduler_mode,
                    demand=demand,
                ),
                reverse=True,
            )

        return self._order_demands_round_robin(state, demands)

    def _order_round_robin(self, state: Any, eligible_ues: list) -> list:
        if len(eligible_ues) == 0:
            return []

        pointer = int(getattr(getattr(state, "scheduler", None), "rr_pointer", 0)) % len(eligible_ues)
        return list(eligible_ues[pointer:]) + list(eligible_ues[:pointer])

    def _get_latency_bias(self, state: Any) -> float:
        scheduler_state = getattr(state, "scheduler", None)
        latency_bias = getattr(scheduler_state, "latency_bias", None)
        if latency_bias is None:
            return 1.0
        return max(0.1, float(latency_bias))

    def _latency_score(self, state: Any, ue: Any, demand: UeSchedulingDemand | None = None) -> float:
        priority_boost_by_ue = getattr(getattr(state, "scheduler", None), "priority_boost_by_ue", {})
        latency_bias = self._get_latency_bias(state)
        hol_delay_ms = float(demand.hol_delay_ms) if demand is not None else 0.0
        backlog_bytes = int(demand.demand_bytes) if demand is not None else int(getattr(ue, "queue_bytes", 0))
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        latency_component = (
            1000.0 / max(float(getattr(ue, "latency_budget_ms", 1.0)), 1.0)
        ) * latency_bias
        hol_component = min(500.0, max(0.0, hol_delay_ms) * (0.5 + 0.5 * latency_bias))
        priority_boost = float(priority_boost_by_ue.get(getattr(ue, "ue_id", ""), 0.0))
        priority_component = (
            100.0
            * (float(getattr(demand, "priority", getattr(ue, "priority", 0)) if demand is not None else getattr(ue, "priority", 0)) + priority_boost)
            * (0.5 + 0.5 * latency_bias)
        )
        queue_component = float(backlog_bytes) / 100.0
        harq_component = (
            float(getattr(demand, "harq_urgency_score", 0.0)) + self.retx_weight_bonus
            if demand is not None and bool(getattr(demand, "has_harq_pending", False))
            else (self.retx_weight_bonus if self._get_pending_retx_bytes(ue) > 0 else 0.0)
        )
        csi_staleness_penalty = min(50.0, float(csi_age_slots) * 5.0)
        confidence_bonus = (
            float(getattr(demand, "channel_quality_confidence", 1.0)) * 10.0
            if demand is not None
            else 0.0
        )
        poor_link_penalty = 0.0
        if self._is_poor_link_probe_candidate(ue, demand, is_retx=False):
            poor_link_penalty = 90.0
        return (
            latency_component
            + hol_component
            + priority_component
            + queue_component
            + harq_component
            + confidence_bonus
            - csi_staleness_penalty
            - poor_link_penalty
        )

    def _slice_score(self, state: Any, ue: Any, demand: UeSchedulingDemand | None = None) -> float:
        slice_weight_map = {
            "urllc": 300.0,
            "embb": 200.0,
            "mmtc": 100.0,
        }
        priority_boost_by_ue = getattr(getattr(state, "scheduler", None), "priority_boost_by_ue", {})
        slice_weight_adjustment = dict(
            getattr(getattr(state, "scheduler", None), "slice_weight_adjustment", {})
        )
        slice_id = str(
            getattr(demand, "slice_id", getattr(ue, "slice_id", "")) if demand is not None else getattr(ue, "slice_id", "")
        ).lower()
        slice_weight = float(slice_weight_adjustment.get(slice_id, 1.0))
        slice_component = slice_weight_map.get(slice_id, 50.0) * max(0.1, slice_weight)
        priority_boost = float(priority_boost_by_ue.get(getattr(ue, "ue_id", ""), 0.0))
        priority_component = 100.0 * (
            float(getattr(demand, "priority", getattr(ue, "priority", 0)) if demand is not None else getattr(ue, "priority", 0))
            + priority_boost
        )
        queue_component = float(
            getattr(demand, "demand_bytes", getattr(ue, "queue_bytes", 0)) if demand is not None else getattr(ue, "queue_bytes", 0)
        ) / 100.0
        radio_component = float(
            getattr(demand, "cqi", getattr(ue, "cqi", getattr(ue, "sinr_db", 0.0))) if demand is not None else getattr(ue, "sinr_db", 0.0)
        )
        harq_component = (
            float(getattr(demand, "harq_urgency_score", 0.0)) + self.retx_weight_bonus
            if demand is not None and bool(getattr(demand, "has_harq_pending", False))
            else (self.retx_weight_bonus if self._get_pending_retx_bytes(ue) > 0 else 0.0)
        )
        csi_staleness_penalty = min(
            50.0,
            float(
                max(
                    int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
                    int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
                    int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
                )
            ) * 5.0,
        )
        poor_link_penalty = 0.0
        if self._is_poor_link_probe_candidate(ue, demand, is_retx=False):
            poor_link_penalty = 75.0
        return (
            slice_component
            + priority_component
            + queue_component
            + radio_component
            + harq_component
            - csi_staleness_penalty
            - poor_link_penalty
        )

    def _score_for_mode(
        self,
        state: Any,
        ue: Any,
        scheduler_mode: str,
        round_robin_rank: int | None = None,
        demand: UeSchedulingDemand | None = None,
    ) -> float:
        if scheduler_mode == "latency_aware":
            return self._latency_score(state, ue, demand=demand)
        if scheduler_mode == "slice_aware":
            return self._slice_score(state, ue, demand=demand)
        if round_robin_rank is None:
            return 1.0
        return float(max(1, self.max_ues_per_slot - round_robin_rank + 1))

    def _apply_policy_constraints(self, state: Any, eligible_ues: list) -> None:
        control = getattr(state, "control", None)
        constraints = list(getattr(control, "active_policy_constraints", [])) if control is not None else []
        if not constraints:
            return

        slot_id = int(getattr(state, "slot_id", -1))

        if not hasattr(getattr(state, "scheduler", None), "priority_boost_by_ue"):
            return
        if not hasattr(getattr(state, "scheduler", None), "last_served_slot_by_ue"):
            return

        eligible_ue_ids = {str(getattr(ue, "ue_id", "")) for ue in eligible_ues}
        last_served_slot_by_ue = state.scheduler.last_served_slot_by_ue

        for constraint in constraints:
            constraint_type = str(getattr(constraint, "constraint_type", ""))
            valid_from_slot = getattr(constraint, "valid_from_slot", None)
            valid_until_slot = getattr(constraint, "valid_until_slot", None)
            if valid_from_slot is not None and slot_id < int(valid_from_slot):
                continue
            if valid_until_slot is not None and slot_id > int(valid_until_slot):
                continue

            if constraint_type != "min_ue_service_interval":
                continue

            params = getattr(constraint, "params", {})
            target_ue_id = getattr(constraint, "target_id", None)
            if target_ue_id is None and isinstance(params, dict):
                target_ue_id = params.get("ue_id")
            if target_ue_id is None:
                continue

            target_ue_id = str(target_ue_id)
            if target_ue_id not in eligible_ue_ids:
                continue

            max_gap_slots = None
            if isinstance(params, dict):
                max_gap_slots = params.get("max_gap_slots")
                if max_gap_slots is None:
                    max_gap_slots = params.get("max_gap")
            if max_gap_slots is None:
                continue

            try:
                max_gap_slots_int = int(max_gap_slots)
            except (TypeError, ValueError):
                continue

            last_served = last_served_slot_by_ue.get(target_ue_id, slot_id)
            gap = slot_id - int(last_served)
            if gap < max_gap_slots_int:
                continue

            current = float(state.scheduler.priority_boost_by_ue.get(target_ue_id, 0.0))
            boost = 100.0
            state.scheduler.priority_boost_by_ue[target_ue_id] = max(current, boost)
            debug_print(
                f"PolicyConstraint consumed at slot {slot_id}: min_ue_service_interval "
                f"ue_id={target_ue_id}, max_gap_slots={max_gap_slots_int}, gap={gap}."
            )

    def _max_grants_allowed(self, eligible_count: int) -> int:
        max_by_prb = self.total_prbs_per_slot // self.min_prbs_per_grant
        max_by_symbol = self.total_symbols_per_slot // self.min_symbols_per_grant
        return max(0, min(int(eligible_count), self.max_ues_per_slot, max_by_prb, max_by_symbol))

    def _allocate_extra_prbs(self, weights: list[float], total_extra_prbs: int) -> list[int]:
        grant_count = len(weights)
        if grant_count <= 0 or total_extra_prbs <= 0:
            return []

        normalized_weights = [max(0.0, float(weight)) for weight in weights]
        if sum(normalized_weights) <= 0.0:
            normalized_weights = [1.0 for _ in range(grant_count)]

        total_weight = float(sum(normalized_weights))
        raw_extras = [int(total_extra_prbs) * weight / total_weight for weight in normalized_weights]
        extra_prbs = [int(raw_value) for raw_value in raw_extras]
        leftover = int(total_extra_prbs) - int(sum(extra_prbs))

        ordering = sorted(
            range(grant_count),
            key=lambda index: (raw_extras[index] - float(extra_prbs[index]), normalized_weights[index]),
            reverse=True,
        )
        for index in ordering[:leftover]:
            extra_prbs[index] += 1

        return extra_prbs

    def _allocate_extra_symbols(self, weights: list[float], total_extra_symbols: int) -> list[int]:
        grant_count = len(weights)
        if grant_count <= 0 or total_extra_symbols <= 0:
            return []

        normalized_weights = [max(0.0, float(weight)) for weight in weights]
        if sum(normalized_weights) <= 0.0:
            normalized_weights = [1.0 for _ in range(grant_count)]

        total_weight = float(sum(normalized_weights))
        raw_extras = [int(total_extra_symbols) * weight / total_weight for weight in normalized_weights]
        extra_symbols = [int(raw_value) for raw_value in raw_extras]
        leftover = int(total_extra_symbols) - int(sum(extra_symbols))

        ordering = sorted(
            range(grant_count),
            key=lambda index: (raw_extras[index] - float(extra_symbols[index]), normalized_weights[index]),
            reverse=True,
        )
        for index in ordering[:leftover]:
            extra_symbols[index] += 1

        return extra_symbols

    def _estimate_base_target_mcs(self, ue: Any) -> int:
        link_quality_db = max(
            float(getattr(ue, "sinr_db", 0.0)),
            float(getattr(ue, "radio_quality_db", 0.0)),
        )
        if link_quality_db < -5.0:
            return 0
        if link_quality_db < -2.0:
            return 2
        if link_quality_db < 1.0:
            return 4
        if link_quality_db < 4.0:
            return 7
        if link_quality_db < 7.0:
            return 10
        if link_quality_db < 10.0:
            return 13
        if link_quality_db < 13.0:
            return 17
        if link_quality_db < 16.0:
            return 20
        if link_quality_db < 19.0:
            return 23
        return 26

    def _derive_target_mcs(
        self,
        ue: Any,
        *,
        demand: UeSchedulingDemand | None = None,
        is_retx: bool,
        scheduler_mode: str,
        allocated_prbs: int,
        allocated_symbols: int,
    ) -> int:
        target_mcs = self._estimate_base_target_mcs(ue)
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        channel_quality_confidence = float(
            getattr(demand, "channel_quality_confidence", 1.0) if demand is not None else 1.0
        )
        if is_retx:
            target_mcs -= 3
        if scheduler_mode == "latency_aware":
            target_mcs -= 2
        if float(getattr(ue, "latency_budget_ms", 0.0)) <= 20.0:
            target_mcs = min(target_mcs, 16)
        if csi_age_slots >= 6:
            target_mcs -= 2
        if channel_quality_confidence < 0.55:
            target_mcs -= 4
        elif channel_quality_confidence < 0.75:
            target_mcs -= 2
        if self._is_poor_link_probe_candidate(ue, demand, is_retx=is_retx):
            target_mcs = min(target_mcs, 4)
        resource_units = int(allocated_prbs) * int(allocated_symbols)
        if (
            scheduler_mode == "slice_aware"
            and str(getattr(ue, "slice_id", "")).lower() == "embb"
            and resource_units >= 120
            and not is_retx
        ):
            target_mcs += 2
        return max(0, min(27, int(target_mcs)))

    def _derive_mcs_cap(self, ue: Any, is_retx: bool, target_mcs: int) -> int | None:
        if is_retx:
            return min(20, int(target_mcs))
        if float(getattr(ue, "latency_budget_ms", 0.0)) <= 20.0:
            return min(16, int(target_mcs))
        return int(target_mcs)

    def _derive_mcs_table_id(
        self,
        ue: Any,
        *,
        demand: UeSchedulingDemand | None,
        is_retx: bool,
        scheduler_mode: str,
        target_mcs: int,
    ) -> str:
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        latency_budget_ms = float(getattr(ue, "latency_budget_ms", 0.0))
        slice_id = str(getattr(ue, "slice_id", "")).strip().lower()
        traffic_type = str(getattr(ue, "traffic_type", "")).strip().lower()
        channel_quality_confidence = float(
            getattr(demand, "channel_quality_confidence", 1.0) if demand is not None else 1.0
        )

        if self._is_poor_link_probe_candidate(ue, demand, is_retx=is_retx):
            return "qam64_low_se"
        if is_retx or latency_budget_ms <= 20.0 or csi_age_slots >= 5 or channel_quality_confidence < 0.55:
            return "qam64_low_se"
        if (
            slice_id == "embb"
            and traffic_type in {"video", "data", "download", "broadband"}
            and scheduler_mode != "latency_aware"
            and int(target_mcs) >= 20
            and csi_age_slots <= 1
            and channel_quality_confidence >= 0.85
        ):
            return "qam256"
        return "qam64"

    def _derive_tbs_selection_method(
        self,
        ue: Any,
        *,
        demand: UeSchedulingDemand | None,
        is_retx: bool,
        scheduler_mode: str,
    ) -> str:
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        latency_budget_ms = float(getattr(ue, "latency_budget_ms", 0.0))
        channel_quality_confidence = float(
            getattr(demand, "channel_quality_confidence", 1.0) if demand is not None else 1.0
        )

        if self._is_poor_link_probe_candidate(ue, demand, is_retx=is_retx):
            return "guarded_resource_elements"
        if is_retx or csi_age_slots >= 6 or channel_quality_confidence < 0.55:
            return "guarded_resource_elements"
        if scheduler_mode == "latency_aware" or latency_budget_ms <= 20.0 or channel_quality_confidence < 0.75:
            return "latency_guarded"
        return "resource_elements"

    def _derive_link_adaptation_reason(
        self,
        ue: Any,
        *,
        demand: UeSchedulingDemand | None,
        is_retx: bool,
        scheduler_mode: str,
        layer_count: int,
        mcs_table_id: str,
        tbs_selection_method: str,
    ) -> str:
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        channel_quality_confidence = float(
            getattr(demand, "channel_quality_confidence", 1.0) if demand is not None else 1.0
        )
        reported_ri = int(
            getattr(demand, "ri", layer_count) if demand is not None else layer_count
        )
        latency_budget_ms = float(getattr(ue, "latency_budget_ms", 0.0))

        if is_retx:
            return "harq_retransmission"
        if self._is_poor_link_probe_candidate(ue, demand, is_retx=is_retx):
            return "poor_link_probe_guarded"
        if channel_quality_confidence < 0.55 and csi_age_slots >= 4:
            return "stale_low_confidence_guarded"
        if channel_quality_confidence < 0.55:
            return "low_confidence_guarded"
        if csi_age_slots >= 5:
            return "stale_csi_guarded"
        if reported_ri > int(layer_count):
            return "rank_limited_adaptation"
        if scheduler_mode == "latency_aware" or latency_budget_ms <= 20.0:
            return "latency_guarded"
        if str(mcs_table_id) == "qam256":
            return "high_confidence_high_se"
        if str(tbs_selection_method) != "resource_elements":
            return f"{str(tbs_selection_method)}_adaptation"
        return "standard_link_adaptation"

    def _derive_dmrs_pattern_id(
        self,
        *,
        demand: UeSchedulingDemand | None,
        is_retx: bool,
        layer_count: int,
    ) -> str:
        csi_age_slots = max(
            int(getattr(demand, "cqi_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "ri_validity_age_slots", 0)) if demand is not None else 0,
            int(getattr(demand, "pmi_validity_age_slots", 0)) if demand is not None else 0,
        )
        if int(layer_count) >= 2 and (is_retx or csi_age_slots >= 4):
            return "dmrs_type1_dense"
        if int(layer_count) >= 2 or is_retx:
            return "dmrs_type1_len2"
        return "dmrs_type1_default"

    def _estimate_grant_bytes_budget(
        self,
        *,
        allocated_prbs: int,
        allocated_symbols: int,
        target_mcs: int,
        requested_bytes: int,
        mcs_table_id: str,
        tbs_selection_method: str,
        dmrs_pattern_id: str | None,
        num_layers: int,
    ) -> int:
        resource_elements = max(0, int(allocated_prbs)) * max(0, int(allocated_symbols)) * 12
        if resource_elements <= 0:
            return 0

        estimated_tb_size_bytes = estimate_tbs_bytes(
            resource_elements=resource_elements,
            mcs_index=int(target_mcs),
            mcs_table_id=normalize_mcs_table_id(mcs_table_id),
            num_layers=max(1, int(num_layers)),
            tbs_selection_method=normalize_tbs_selection_method(tbs_selection_method),
            dmrs_pattern_id=dmrs_pattern_id,
        )
        return min(max(0, int(requested_bytes)), estimated_tb_size_bytes)

    def _build_resource_block_indices(
        self,
        *,
        start_index: int,
        allocation_size: int,
        allocation_limit: int,
    ) -> list[int]:
        normalized_start = max(0, int(start_index))
        normalized_size = max(0, int(allocation_size))
        normalized_limit = max(0, int(allocation_limit))
        allocation_end = min(normalized_limit, normalized_start + normalized_size)
        return list(range(normalized_start, max(normalized_start, allocation_end)))

    def _pack_grants_frequency_first(self, grants: list[UeGrant]) -> None:
        next_start_prb = 0
        for grant in grants:
            grant.start_prb = int(next_start_prb)
            grant.start_symbol = 0
            grant.resource_block_indices = self._build_resource_block_indices(
                start_index=grant.start_prb,
                allocation_size=int(grant.allocated_prbs),
                allocation_limit=int(self.total_prbs_per_slot),
            )
            grant.ofdm_symbol_indices = self._build_resource_block_indices(
                start_index=grant.start_symbol,
                allocation_size=int(grant.allocated_symbols),
                allocation_limit=int(self.total_symbols_per_slot),
            )
            next_start_prb += int(grant.allocated_prbs)

    def _pack_grants_time_first(self, grants: list[UeGrant]) -> None:
        next_start_symbol = 0
        for grant in grants:
            grant.start_prb = 0
            grant.start_symbol = int(next_start_symbol)
            grant.resource_block_indices = self._build_resource_block_indices(
                start_index=grant.start_prb,
                allocation_size=int(grant.allocated_prbs),
                allocation_limit=int(self.total_prbs_per_slot),
            )
            grant.ofdm_symbol_indices = self._build_resource_block_indices(
                start_index=grant.start_symbol,
                allocation_size=int(grant.allocated_symbols),
                allocation_limit=int(self.total_symbols_per_slot),
            )
            next_start_symbol += int(grant.allocated_symbols)

    def _apply_packing_strategy(self, state: Any, grants: list[UeGrant]) -> str:
        strategy = self._get_packing_strategy(state)
        if strategy == "time_first":
            self._pack_grants_time_first(grants)
            return "time_first"

        self._pack_grants_frequency_first(grants)
        return "frequency_first"

    def _apply_anchor_resource_shape(self, grant: UeGrant, anchor_grant: UeGrant) -> None:
        grant.allocated_prbs = min(int(grant.allocated_prbs), int(anchor_grant.allocated_prbs))
        grant.allocated_symbols = min(int(grant.allocated_symbols), int(anchor_grant.allocated_symbols))
        grant.start_prb = int(anchor_grant.start_prb)
        grant.start_symbol = int(anchor_grant.start_symbol)
        grant.resource_block_indices = self._build_resource_block_indices(
            start_index=grant.start_prb,
            allocation_size=int(grant.allocated_prbs),
            allocation_limit=int(self.total_prbs_per_slot),
        )
        grant.ofdm_symbol_indices = self._build_resource_block_indices(
            start_index=grant.start_symbol,
            allocation_size=int(grant.allocated_symbols),
            allocation_limit=int(self.total_symbols_per_slot),
        )

    def _apply_spatial_multiplexing(
        self,
        state: Any,
        grants: list[UeGrant],
        ue_map: dict[str, Any],
    ) -> None:
        if not self._get_spatial_multiplexing_enabled(state) or len(grants) < 2:
            return

        max_mu_users_per_group = self._get_max_mu_users_per_group(state)
        max_total_spatial_layers = self._get_max_total_spatial_layers(state)
        next_group_id = 0
        groups: list[dict[str, Any]] = []

        for grant in grants:
            ue = ue_map.get(str(grant.ue_id))
            if ue is None or bool(getattr(grant, "is_retx", False)):
                continue

            candidate_layers = self._get_layer_count(state, ue)
            best_group: dict[str, Any] | None = None
            best_score = 0.0
            for group in groups:
                if int(group["member_count"]) >= max_mu_users_per_group:
                    continue
                if int(group["total_layers"]) + int(candidate_layers) > max_total_spatial_layers:
                    continue

                compatibility_scores = [
                    self._compute_spatial_compatibility_score(state, ue, member_ue)
                    for member_ue in group["member_ues"]
                ]
                compatibility_score = min(compatibility_scores) if compatibility_scores else 0.0
                if compatibility_score <= 0.0:
                    continue
                if compatibility_score > best_score:
                    best_score = compatibility_score
                    best_group = group

            if best_group is None:
                groups.append(
                    {
                        "anchor_grant": grant,
                        "member_ues": [ue],
                        "member_count": 1,
                        "total_layers": int(candidate_layers),
                        "group_id": None,
                    }
                )
                continue

            if best_group["group_id"] is None:
                best_group["group_id"] = next_group_id
                next_group_id += 1
                anchor_grant = best_group["anchor_grant"]
                anchor_grant.spatial_group_id = int(best_group["group_id"])
                anchor_grant.spatial_group_size = 2
                anchor_grant.spatial_compatibility_score = max(
                    float(anchor_grant.spatial_compatibility_score),
                    float(best_score),
                )
            else:
                anchor_grant = best_group["anchor_grant"]
                anchor_grant.spatial_group_size = int(best_group["member_count"]) + 1
                anchor_grant.spatial_compatibility_score = max(
                    float(anchor_grant.spatial_compatibility_score),
                    float(best_score),
                )

            grant.spatial_group_id = int(best_group["group_id"])
            grant.spatial_group_size = int(best_group["member_count"]) + 1
            grant.spatial_compatibility_score = float(best_score)
            self._apply_anchor_resource_shape(grant, anchor_grant)
            grant.grant_bytes_budget = self._estimate_grant_bytes_budget(
                allocated_prbs=int(grant.allocated_prbs),
                allocated_symbols=int(grant.allocated_symbols),
                target_mcs=int(grant.target_mcs or 0),
                requested_bytes=int(grant.requested_bytes),
                mcs_table_id=str(getattr(grant, "mcs_table_id", "qam64")),
                tbs_selection_method=str(getattr(grant, "tbs_selection_method", "resource_elements")),
                dmrs_pattern_id=getattr(grant, "dmrs_pattern_id", None),
                num_layers=self._get_layer_count(state, ue),
            )

            best_group["member_ues"].append(ue)
            best_group["member_count"] = int(best_group["member_count"]) + 1
            best_group["total_layers"] = int(best_group["total_layers"]) + int(candidate_layers)
            for member_grant in grants:
                if member_grant.spatial_group_id == grant.spatial_group_id:
                    member_grant.spatial_group_size = int(best_group["member_count"])
                    member_grant.spatial_compatibility_score = max(
                        float(member_grant.spatial_compatibility_score),
                        float(best_score),
                    )

    def _estimate_unique_resource_usage(self, grants: list[UeGrant]) -> tuple[int, int]:
        unique_prbs: set[int] = set()
        unique_symbols: set[int] = set()
        for grant in grants:
            unique_prbs.update(int(item) for item in list(getattr(grant, "resource_block_indices", []) or []))
            unique_symbols.update(int(item) for item in list(getattr(grant, "ofdm_symbol_indices", []) or []))
        return len(unique_prbs), len(unique_symbols)

    def _build_decision(
        self,
        state: Any,
        selected_retx_demands: list[UeSchedulingDemand],
        selected_new_demands: list[UeSchedulingDemand],
        scheduler_mode: str,
        *,
        ue_map: dict[str, Any],
        demands: list[UeSchedulingDemand],
        stage_trace: SchedulingStageTrace,
    ) -> SchedulingDecision:
        selected_demands = list(selected_retx_demands) + list(selected_new_demands)
        if len(selected_demands) == 0:
            return SchedulingDecision(
                slot_id=int(getattr(state, "slot_id", -1)),
                scheduler_mode=str(scheduler_mode),
                total_prbs=int(self.total_prbs_per_slot),
                total_symbols=int(self.total_symbols_per_slot),
                grants=[],
                demands=list(demands),
                stage_trace=stage_trace,
                unallocated_prbs=int(self.total_prbs_per_slot),
                unallocated_symbols=int(self.total_symbols_per_slot),
            )

        round_robin_order = {
            str(getattr(demand, "ue_id", "")): index
            for index, demand in enumerate(selected_demands)
        }
        grant_specs = []
        for demand in selected_demands:
            ue = ue_map.get(str(demand.ue_id))
            if ue is None:
                continue
            is_retx = bool(demand.has_harq_pending)
            if is_retx:
                active_harq_process = self._get_active_harq_process(ue)
                harq_process_id = (
                    int(
                        getattr(
                            active_harq_process,
                            "process_id",
                            getattr(getattr(ue, "harq", None), "process_id", 0),
                        )
                    )
                    if active_harq_process is not None
                    else int(getattr(getattr(ue, "harq", None), "process_id", 0))
                )
                harq_redundancy_version = (
                    int(
                        getattr(
                            active_harq_process,
                            "redundancy_version",
                            getattr(getattr(ue, "harq", None), "redundancy_version", 0),
                        )
                    )
                    if active_harq_process is not None
                    else int(getattr(getattr(ue, "harq", None), "redundancy_version", 0))
                )
                harq_ndi = (
                    int(
                        getattr(
                            active_harq_process,
                            "ndi",
                            getattr(getattr(ue, "harq", None), "ndi", 0),
                        )
                    )
                    if active_harq_process is not None
                    else int(getattr(getattr(ue, "harq", None), "ndi", 0))
                )
                harq_retransmission_round = (
                    int(
                        getattr(
                            active_harq_process,
                            "retransmission_round",
                            getattr(getattr(ue, "harq", None), "retransmission_count", 0),
                        )
                    )
                    if active_harq_process is not None
                    else int(getattr(getattr(ue, "harq", None), "retransmission_count", 0))
                )
            else:
                (
                    harq_process_id,
                    harq_redundancy_version,
                    harq_ndi,
                    harq_retransmission_round,
                ) = self._reserve_new_tx_harq_context(state, ue, demand)
            score = self._score_for_mode(
                state,
                ue,
                scheduler_mode,
                round_robin_rank=round_robin_order.get(str(getattr(demand, "ue_id", "")), 0),
                demand=demand,
            )
            if is_retx:
                score += float(self.retx_weight_bonus) + float(demand.harq_urgency_score)
            poor_link_probe = self._is_poor_link_probe_candidate(ue, demand, is_retx=bool(is_retx))
            requested_bytes = int(demand.requested_bytes)
            if poor_link_probe:
                requested_bytes = min(
                    int(requested_bytes),
                    int(self._derive_poor_link_probe_requested_bytes_cap(ue, demand)),
                )
                score *= float(self._poor_link_probe_score_scale(ue, demand))
            byte_weight = float(max(0, int(requested_bytes))) / 100.0
            selection_reason = stage_trace.selection_reasons_by_ue.get(
                str(getattr(demand, "ue_id", "")),
                self._build_selection_reason(
                    scheduler_mode=scheduler_mode,
                    demand=demand,
                    is_retx=bool(is_retx),
                ),
            )
            resource_pressure_source = stage_trace.resource_pressure_source_by_ue.get(
                str(getattr(demand, "ue_id", "")),
                self._build_resource_pressure_source(
                    is_retx=bool(is_retx),
                    total_candidate_count=len(demands),
                    selected_retx_count=len(selected_retx_demands),
                ),
            )
            constraint_hits = list(
                stage_trace.constraint_hits_by_ue.get(
                    str(getattr(demand, "ue_id", "")),
                    self._build_constraint_hits(state, ue, demand, is_retx=bool(is_retx)),
                )
            )
            grant_specs.append(
                {
                    "ue": ue,
                    "demand": demand,
                    "is_retx": bool(is_retx),
                    "requested_bytes": int(requested_bytes),
                    "priority_score": max(1.0, float(score) + byte_weight),
                    "harq_process_id": int(harq_process_id),
                    "harq_redundancy_version": int(harq_redundancy_version),
                    "harq_ndi": int(harq_ndi),
                    "harq_retransmission_round": int(harq_retransmission_round),
                    "selection_reason": str(selection_reason),
                    "resource_pressure_source": str(resource_pressure_source),
                    "constraint_hits": list(constraint_hits),
                    "poor_link_probe": bool(poor_link_probe),
                }
            )

        grant_count = len(grant_specs)
        base_prb_allocations = [int(self.min_prbs_per_grant) for _ in range(grant_count)]
        base_symbol_allocations = [int(self.min_symbols_per_grant) for _ in range(grant_count)]
        extra_prb_budget = max(0, int(self.total_prbs_per_slot) - int(sum(base_prb_allocations)))
        extra_symbol_budget = max(0, int(self.total_symbols_per_slot) - int(sum(base_symbol_allocations)))
        weights = [float(spec["priority_score"]) for spec in grant_specs]
        extra_prb_allocations = self._allocate_extra_prbs(weights, extra_prb_budget)
        extra_symbol_allocations = self._allocate_extra_symbols(weights, extra_symbol_budget)

        grants: list[UeGrant] = []
        prb_extras = extra_prb_allocations if extra_prb_allocations else [0 for _ in range(grant_count)]
        symbol_extras = extra_symbol_allocations if extra_symbol_allocations else [0 for _ in range(grant_count)]
        for spec, base_prbs, extra_prbs, base_symbols, extra_symbols in zip(
            grant_specs,
            base_prb_allocations,
            prb_extras,
            base_symbol_allocations,
            symbol_extras,
        ):
            ue = spec["ue"]
            allocated_prbs = int(base_prbs) + int(extra_prbs)
            allocated_symbols = int(base_symbols) + int(extra_symbols)
            target_mcs = self._derive_target_mcs(
                ue,
                demand=spec["demand"],
                is_retx=bool(spec["is_retx"]),
                scheduler_mode=scheduler_mode,
                allocated_prbs=allocated_prbs,
                allocated_symbols=allocated_symbols,
            )
            layer_count = self._get_layer_count(state, ue)
            mcs_table_id = self._derive_mcs_table_id(
                ue,
                demand=spec["demand"],
                is_retx=bool(spec["is_retx"]),
                scheduler_mode=scheduler_mode,
                target_mcs=target_mcs,
            )
            tbs_selection_method = self._derive_tbs_selection_method(
                ue,
                demand=spec["demand"],
                is_retx=bool(spec["is_retx"]),
                scheduler_mode=scheduler_mode,
            )
            dmrs_pattern_id = self._derive_dmrs_pattern_id(
                demand=spec["demand"],
                is_retx=bool(spec["is_retx"]),
                layer_count=layer_count,
            )
            channel_quality_confidence = float(
                getattr(spec["demand"], "channel_quality_confidence", 1.0)
            )
            link_adaptation_reason = self._derive_link_adaptation_reason(
                ue,
                demand=spec["demand"],
                is_retx=bool(spec["is_retx"]),
                scheduler_mode=scheduler_mode,
                layer_count=layer_count,
                mcs_table_id=mcs_table_id,
                tbs_selection_method=tbs_selection_method,
            )
            grant_bytes_budget = self._estimate_grant_bytes_budget(
                allocated_prbs=allocated_prbs,
                allocated_symbols=allocated_symbols,
                target_mcs=target_mcs,
                requested_bytes=int(spec["requested_bytes"]),
                mcs_table_id=mcs_table_id,
                tbs_selection_method=tbs_selection_method,
                dmrs_pattern_id=dmrs_pattern_id,
                num_layers=layer_count,
            )
            grants.append(
                UeGrant(
                    ue_id=str(getattr(ue, "ue_id", "unknown_ue")),
                    allocated_prbs=allocated_prbs,
                    allocated_symbols=allocated_symbols,
                    dmrs_pattern_id=dmrs_pattern_id,
                    is_retx=bool(spec["is_retx"]),
                    requested_bytes=int(spec["requested_bytes"]),
                    grant_bytes_budget=int(grant_bytes_budget),
                    priority_score=float(spec["priority_score"]),
                    target_mcs=int(target_mcs),
                    mcs_cap=self._derive_mcs_cap(ue, bool(spec["is_retx"]), target_mcs),
                    mcs_table_id=mcs_table_id,
                    tbs_selection_method=tbs_selection_method,
                    harq_process_id=int(spec["harq_process_id"]),
                    harq_redundancy_version=int(spec["harq_redundancy_version"]),
                    harq_ndi=int(spec["harq_ndi"]),
                    harq_retransmission_round=int(spec["harq_retransmission_round"]),
                    cqi_validity_age_slots=int(getattr(spec["demand"], "cqi_validity_age_slots", 0)),
                    ri_validity_age_slots=int(getattr(spec["demand"], "ri_validity_age_slots", 0)),
                    pmi_validity_age_slots=int(getattr(spec["demand"], "pmi_validity_age_slots", 0)),
                    channel_quality_confidence=float(channel_quality_confidence),
                    link_adaptation_reason=str(link_adaptation_reason),
                    selection_reason=str(spec["selection_reason"]),
                    resource_pressure_source=str(spec["resource_pressure_source"]),
                    harq_preemption_cause=stage_trace.harq_preemption_causes_by_ue.get(str(getattr(ue, "ue_id", ""))),
                    constraint_hits=list(spec["constraint_hits"]),
                )
            )

        applied_packing_strategy = self._apply_packing_strategy(state, grants)
        try:
            state.scheduler.packing_strategy = applied_packing_strategy
            state.scheduler.enable_spatial_multiplexing = self._get_spatial_multiplexing_enabled(state)
        except Exception:
            pass

        self._apply_spatial_multiplexing(state, grants, ue_map)
        self._assign_logical_channel_allocations(
            grants,
            ue_map,
            demand_map={str(demand.ue_id): demand for demand in demands},
        )
        for grant in grants:
            summary = self._summarize_logical_channel_allocations(grant)
            grant.lc_multiplexing_summary = dict(summary)
            stage_trace.lc_multiplexing_summaries_by_ue[str(grant.ue_id)] = dict(summary)

        allocated_prbs_total, allocated_symbols_total = self._estimate_unique_resource_usage(grants)
        stage_trace.grant_candidate_ue_ids = [str(getattr(demand, "ue_id", "")) for demand in selected_demands]
        stage_trace.granted_ue_ids = [str(grant.ue_id) for grant in grants if grant.is_active]
        return SchedulingDecision(
            slot_id=int(getattr(state, "slot_id", -1)),
            scheduler_mode=str(scheduler_mode),
            total_prbs=int(self.total_prbs_per_slot),
            total_symbols=int(self.total_symbols_per_slot),
            grants=grants,
            demands=list(demands),
            stage_trace=stage_trace,
            unallocated_prbs=max(0, int(self.total_prbs_per_slot) - int(allocated_prbs_total)),
            unallocated_symbols=max(0, int(self.total_symbols_per_slot) - int(allocated_symbols_total)),
        )

    def _write_decision_to_state(self, state: Any, decision: SchedulingDecision) -> None:
        state.scheduler.grants = list(decision.grants)
        state.scheduler.last_decision = decision
        try:
            state.scheduler.last_demands = list(getattr(decision, "demands", []) or [])
            state.scheduler.last_stage_trace = getattr(decision, "stage_trace", None)
        except Exception:
            pass
        state.scheduler.scheduled_ue_ids = list(decision.scheduled_ue_ids)

        grant_map = {str(grant.ue_id): grant for grant in decision.grants}
        for ue in getattr(state, "ues", []):
            ue_id = str(getattr(ue, "ue_id", "unknown_ue"))
            grant = grant_map.get(ue_id)
            connection = self._get_connection(state, ue_id)
            try:
                ue.is_scheduled = grant is not None and grant.is_active
                ue.allocated_prbs = int(getattr(grant, "allocated_prbs", 0)) if grant is not None else 0
                ue.allocated_symbols = int(getattr(grant, "allocated_symbols", 0)) if grant is not None else 0
                ue.start_prb = int(getattr(grant, "start_prb", 0)) if grant is not None else 0
                ue.start_symbol = int(getattr(grant, "start_symbol", 0)) if grant is not None else 0
                ue.is_retx = bool(getattr(grant, "is_retx", False)) if grant is not None else False
                ue.priority_score = float(getattr(grant, "priority_score", 0.0)) if grant is not None else 0.0
                ue.grant_bytes_budget = int(getattr(grant, "grant_bytes_budget", 0)) if grant is not None else 0
                ue.grant_target_mcs = getattr(grant, "target_mcs", None) if grant is not None else None
                ue.resource_block_indices = list(getattr(grant, "resource_block_indices", []) or []) if grant is not None else []
                ue.allocated_symbol_indices = list(getattr(grant, "ofdm_symbol_indices", []) or []) if grant is not None else []
                ue.spatial_group_id = getattr(grant, "spatial_group_id", None) if grant is not None else None
                ue.spatial_group_size = int(getattr(grant, "spatial_group_size", 1)) if grant is not None else 1
                ue.spatial_compatibility_score = float(getattr(grant, "spatial_compatibility_score", 0.0)) if grant is not None else 0.0
                ue.spatial_mode = "mu_mimo" if grant is not None and getattr(grant, "spatial_group_id", None) is not None else "su_siso"
                ue.grant_selection_reason = str(getattr(grant, "selection_reason", "")) if grant is not None else ""
                ue.grant_rejection_reason = (
                    str(getattr(getattr(decision, "stage_trace", None), "rejection_reasons_by_ue", {}).get(ue_id, ""))
                )
                ue.grant_resource_pressure_source = str(getattr(grant, "resource_pressure_source", "")) if grant is not None else str(
                    getattr(getattr(decision, "stage_trace", None), "resource_pressure_source_by_ue", {}).get(ue_id, "")
                )
                ue.grant_harq_preemption_cause = (
                    getattr(grant, "harq_preemption_cause", None)
                    if grant is not None
                    else getattr(getattr(decision, "stage_trace", None), "harq_preemption_causes_by_ue", {}).get(ue_id)
                )
                ue.grant_constraint_hits = (
                    list(getattr(grant, "constraint_hits", []) or [])
                    if grant is not None
                    else list(getattr(getattr(decision, "stage_trace", None), "constraint_hits_by_ue", {}).get(ue_id, []) or [])
                )
                ue.grant_lc_multiplexing_summary = (
                    dict(getattr(grant, "lc_multiplexing_summary", {}) or {})
                    if grant is not None
                    else {}
                )
                ue.scheduled_logical_channel_ids = [
                    str(getattr(allocation, "logical_channel_id", ""))
                    for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                    if str(getattr(allocation, "logical_channel_id", "")) != ""
                ] if grant is not None else []
                ue.scheduled_drb_ids = [
                    int(getattr(allocation, "drb_id", 0))
                    for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                    if getattr(allocation, "drb_id", None) is not None
                ] if grant is not None else []
                ue.scheduled_lcg_ids = sorted(
                    {
                        int(getattr(allocation, "lcg_id", 0))
                        for allocation in list(getattr(grant, "logical_channel_allocations", []) or [])
                        if getattr(allocation, "lcg_id", None) is not None
                    }
                ) if grant is not None else []
            except Exception:
                pass
            if connection is not None and grant is not None and bool(getattr(grant, "is_active", True)):
                try:
                    connection.last_scheduled_slot = int(getattr(state, "slot_id", -1))
                    connection.serving_beam_id = int(
                        getattr(ue, "serving_beam_id", getattr(connection, "serving_beam_id", 0))
                    )
                except Exception:
                    pass

    def run(self, state: Any) -> List[str]:
        """
        Runs one scheduling decision for the current slot.
        INPUT
        state: The global simulation state.
        OUTPUT
        A list containing the scheduled UE identifiers for the current slot.
        EXAMPLE OUTPUT
        ["ue_1", "ue_2"]
        """
        if len(getattr(state, "ues", [])) == 0:
            self._clear_scheduler_outputs(state)
            return []

        self._refresh_logical_channel_pbr_credits(state)
        demands, ue_map, stage_trace = self._build_scheduling_demands(state)
        if len(demands) == 0:
            self._clear_scheduler_outputs(state, demands=[], stage_trace=stage_trace)
            return []

        eligible_ues = [
            ue_map[str(demand.ue_id)]
            for demand in demands
            if str(demand.ue_id) in ue_map
        ]

        self._apply_policy_constraints(state, eligible_ues)

        scheduler_mode = str(getattr(getattr(state, "scheduler", None), "scheduler_mode", "round_robin"))
        max_grants = self._max_grants_allowed(len(demands))
        if max_grants <= 0:
            for demand in demands:
                stage_trace.filtered_out_reasons.setdefault(str(demand.ue_id), "no_slot_budget")
            stage_trace.filtered_out_ue_ids = sorted(
                set(stage_trace.filtered_out_ue_ids)
                | {str(demand.ue_id) for demand in demands}
            )
            self._clear_scheduler_outputs(state, demands=demands, stage_trace=stage_trace)
            return []

        retx_demands = [demand for demand in demands if bool(getattr(demand, "has_harq_pending", False))]
        retx_ue_ids = {str(demand.ue_id) for demand in retx_demands}
        new_tx_demands: list[UeSchedulingDemand] = []
        for demand in demands:
            ue_id = str(demand.ue_id)
            if ue_id in retx_ue_ids:
                continue
            ue = ue_map.get(ue_id)
            harq = getattr(ue, "harq", None) if ue is not None else None
            if harq is not None and not self._has_available_harq_process(harq):
                stage_trace.filtered_out_reasons.setdefault(ue_id, "no_free_harq_process")
                continue
            new_tx_demands.append(demand)
        stage_trace.filtered_out_ue_ids = sorted(stage_trace.filtered_out_reasons.keys())
        stage_trace.retx_candidate_ue_ids = [str(demand.ue_id) for demand in retx_demands]
        stage_trace.new_tx_candidate_ue_ids = [str(demand.ue_id) for demand in new_tx_demands]

        ordered_retx = self._order_demands_for_mode(state, retx_demands, ue_map, scheduler_mode)
        ordered_new = self._order_demands_for_mode(state, new_tx_demands, ue_map, scheduler_mode)

        selected_retx = list(ordered_retx[:max_grants])
        remaining_slots = max_grants - len(selected_retx)
        selected_new: list[UeSchedulingDemand] = []
        if remaining_slots > 0:
            selected_new = list(ordered_new[:remaining_slots])
        stage_trace.selected_retx_ue_ids = [str(demand.ue_id) for demand in selected_retx]
        stage_trace.selected_new_tx_ue_ids = [str(demand.ue_id) for demand in selected_new]
        selected_demands = list(selected_retx) + list(selected_new)
        selected_ue_ids = {str(demand.ue_id) for demand in selected_demands}

        for demand in demands:
            ue_id = str(demand.ue_id)
            ue = ue_map.get(ue_id)
            if ue is None:
                continue
            is_retx = bool(getattr(demand, "has_harq_pending", False))
            constraint_hits = self._build_constraint_hits(
                state,
                ue,
                demand,
                is_retx=is_retx,
            )
            if len(constraint_hits) > 0:
                stage_trace.constraint_hits_by_ue[ue_id] = list(constraint_hits)
            if ue_id in selected_ue_ids:
                stage_trace.selection_reasons_by_ue[ue_id] = self._build_selection_reason(
                    scheduler_mode=scheduler_mode,
                    demand=demand,
                    is_retx=is_retx,
                )
                stage_trace.resource_pressure_source_by_ue[ue_id] = self._build_resource_pressure_source(
                    is_retx=is_retx,
                    total_candidate_count=len(demands),
                    selected_retx_count=len(selected_retx),
                )
                continue
            if ue_id in stage_trace.filtered_out_reasons:
                continue
            rejection_reason = self._build_rejection_reason(
                scheduler_mode=scheduler_mode,
                demand=demand,
                is_retx=is_retx,
                selected_retx_count=len(selected_retx),
                remaining_new_tx_slots=remaining_slots,
            )
            stage_trace.rejection_reasons_by_ue[ue_id] = rejection_reason
            stage_trace.filtered_out_reasons.setdefault(ue_id, rejection_reason)
            if ue_id not in stage_trace.filtered_out_ue_ids:
                stage_trace.filtered_out_ue_ids.append(ue_id)
            stage_trace.resource_pressure_source_by_ue[ue_id] = (
                "harq_retx_budget" if (not is_retx and len(selected_retx) > 0 and remaining_slots <= 0)
                else "shared_slot_budget"
            )
            if not is_retx and len(selected_retx) > 0 and remaining_slots <= 0:
                stage_trace.harq_preemption_causes_by_ue[ue_id] = "retx_reserved_slot_budget"

        stage_trace.filtered_out_ue_ids = sorted(set(stage_trace.filtered_out_ue_ids))

        if scheduler_mode == "round_robin" and len(demands) > 0:
            advanced = len(
                [
                    demand
                    for demand in selected_demands
                    if str(getattr(demand, "ue_id", "")) not in retx_ue_ids
                ]
            )
            if advanced > 0:
                state.scheduler.rr_pointer = (int(state.scheduler.rr_pointer) + advanced) % len(demands)

        decision = self._build_decision(
            state,
            selected_retx,
            selected_new,
            scheduler_mode,
            ue_map=ue_map,
            demands=demands,
            stage_trace=stage_trace,
        )
        self._write_decision_to_state(state, decision)

        try:
            slot_id = int(getattr(state, "slot_id", -1))
            if hasattr(getattr(state, "scheduler", None), "last_served_slot_by_ue"):
                for ue_id in state.scheduler.scheduled_ue_ids:
                    state.scheduler.last_served_slot_by_ue[str(ue_id)] = slot_id
        except Exception:
            pass

        grant_summary = ", ".join(
            f"{grant.ue_id}:{grant.allocated_prbs}prb/{grant.allocated_symbols}sym"
            f"@p{grant.start_prb}/s{grant.start_symbol}"
            f"/mcs{grant.target_mcs}/tb{grant.grant_bytes_budget}B"
            f"/lc{len(getattr(grant, 'logical_channel_allocations', []) or [])}"
            f"{'/mu' + str(grant.spatial_group_id) if grant.spatial_group_id is not None else ''}"
            f"{'-retx' if grant.is_retx else ''}"
            for grant in decision.grants
        )
        debug_print(
            f"Scheduler mode={scheduler_mode}, packing={getattr(state.scheduler, 'packing_strategy', 'frequency_first')} "
            f"produced grants [{grant_summary}] "
            f"at slot {getattr(state, 'slot_id', -1)}."
        )

        try:
            state.scheduler.priority_boost_by_ue = {}
        except Exception:
            pass

        return state.scheduler.scheduled_ue_ids
