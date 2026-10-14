"""models/scheduling.py

Grant-based MAC scheduling data models.

The simulator previously represented scheduling as only a list of selected
UE identifiers. This module upgrades that abstraction so one slot can carry
multiple per-UE grants with explicit PRB/symbol allocations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LogicalChannelGrant:
    """One logical-channel allocation within one UE grant."""

    logical_channel_id: str
    drb_id: int | None = None
    lcg_id: int | None = None
    scheduled_bytes: int = 0
    queue_bytes_before: int = 0
    priority: int = 0
    packet_delay_budget_ms: float = 0.0
    slice_id: str = ""
    traffic_type: str = ""
    tx_contract: object | None = None
    is_retx: bool = False
    prioritized_bytes: int = 0
    best_effort_bytes: int = 0
    used_pbr_credit_bytes: int = 0
    allocation_reason: str = ""
    constraint_hits: list[str] = field(default_factory=list)


@dataclass
class LogicalChannelDemand:
    """One logical-channel demand object consumed by the scheduler."""

    logical_channel_id: str
    drb_id: int | None = None
    lcg_id: int | None = None
    true_backlog_bytes: int = 0
    observed_backlog_bytes: int = 0
    demand_bytes: int = 0
    backlog_bytes: int = 0
    hol_delay_ms: float = 0.0
    priority: int = 0
    packet_delay_budget_ms: float = 0.0
    qos_class: str = ""
    five_qi: int | None = None
    slice_id: str = ""
    traffic_type: str = ""
    bsr_like_bytes: int = 0
    bsr_reported_bytes: int = 0
    bsr_pending_bytes: int = 0
    bsr_age_slots: int = 0
    measurement_age_slots: int = 0
    pbr_credit_bytes: float = 0.0


@dataclass
class LogicalChannelGroupDemand:
    """One logical-channel-group demand object consumed by the scheduler."""

    lcg_id: int
    logical_channel_ids: list[str] = field(default_factory=list)
    drb_ids: list[int] = field(default_factory=list)
    true_backlog_bytes: int = 0
    observed_backlog_bytes: int = 0
    demand_bytes: int = 0
    prioritized_queue_bytes: int = 0
    dominant_priority: int = 0
    min_packet_delay_budget_ms: float = 0.0
    qos_class: str = ""
    five_qi: int | None = None
    bsr_reported_bytes: int = 0
    bsr_pending_bytes: int = 0
    bsr_age_slots: int = 0
    bsr_triggered: bool = False
    pbr_credit_bytes: float = 0.0


@dataclass
class UeSchedulingDemand:
    """One UE-level demand object consumed by staged scheduling."""

    ue_id: str
    true_backlog_bytes: int = 0
    demand_bytes: int = 0
    backlog_bytes: int = 0
    observed_backlog_bytes: int = 0
    hol_delay_ms: float = 0.0
    priority: int = 0
    qos_class: str = ""
    five_qi: int | None = None
    slice_id: str = ""
    traffic_type: str = ""
    bsr_like_bytes: int = 0
    bsr_reported_bytes: int = 0
    bsr_pending_bytes: int = 0
    harq_pending_bytes: int = 0
    harq_urgency_score: float = 0.0
    cqi: int = 0
    ri: int = 1
    pmi: int = 0
    cqi_validity_age_slots: int = 0
    ri_validity_age_slots: int = 0
    pmi_validity_age_slots: int = 0
    measurement_age_slots: int = 0
    channel_quality_confidence: float = 1.0
    connection_state: str = "connected"
    demand_source: str = "mac_state"
    demand_reason: str = "buffered_traffic"
    logical_channel_demands: list[LogicalChannelDemand] = field(default_factory=list)
    logical_channel_group_demands: list[LogicalChannelGroupDemand] = field(default_factory=list)

    @property
    def has_harq_pending(self) -> bool:
        return int(self.harq_pending_bytes) > 0

    @property
    def requested_bytes(self) -> int:
        if self.has_harq_pending:
            return int(self.harq_pending_bytes)
        return int(self.demand_bytes if int(self.demand_bytes) > 0 else self.backlog_bytes)


@dataclass
class SchedulingStageTrace:
    """Captures staged scheduler flow for one slot."""

    demand_source: str = "mac_state"
    demand_visible_ue_ids: list[str] = field(default_factory=list)
    low_confidence_channel_ue_ids: list[str] = field(default_factory=list)
    stale_csi_ue_ids: list[str] = field(default_factory=list)
    harq_urgent_ue_ids: list[str] = field(default_factory=list)
    feasible_candidate_ue_ids: list[str] = field(default_factory=list)
    filtered_out_ue_ids: list[str] = field(default_factory=list)
    filtered_out_reasons: dict[str, str] = field(default_factory=dict)
    retx_candidate_ue_ids: list[str] = field(default_factory=list)
    new_tx_candidate_ue_ids: list[str] = field(default_factory=list)
    selected_retx_ue_ids: list[str] = field(default_factory=list)
    selected_new_tx_ue_ids: list[str] = field(default_factory=list)
    grant_candidate_ue_ids: list[str] = field(default_factory=list)
    granted_ue_ids: list[str] = field(default_factory=list)
    selection_reasons_by_ue: dict[str, str] = field(default_factory=dict)
    rejection_reasons_by_ue: dict[str, str] = field(default_factory=dict)
    resource_pressure_source_by_ue: dict[str, str] = field(default_factory=dict)
    harq_preemption_causes_by_ue: dict[str, str] = field(default_factory=dict)
    constraint_hits_by_ue: dict[str, list[str]] = field(default_factory=dict)
    lc_multiplexing_summaries_by_ue: dict[str, dict[str, object]] = field(default_factory=dict)
    per_cell_stage_trace: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class UeGrant:
    """One UE grant within a slot scheduling decision."""

    ue_id: str
    allocated_prbs: int
    allocated_symbols: int
    start_prb: int = 0
    start_symbol: int = 0
    resource_block_indices: list[int] = field(default_factory=list)
    ofdm_symbol_indices: list[int] = field(default_factory=list)
    dmrs_pattern_id: str | None = None
    is_retx: bool = False
    requested_bytes: int = 0
    grant_bytes_budget: int = 0
    priority_score: float = 0.0
    target_mcs: int | None = None
    mcs_cap: int | None = None
    mcs_table_id: str = "qam64"
    tbs_selection_method: str = "resource_elements"
    harq_process_id: int | None = None
    harq_redundancy_version: int | None = None
    harq_ndi: int | None = None
    harq_retransmission_round: int = 0
    cqi_validity_age_slots: int = 0
    ri_validity_age_slots: int = 0
    pmi_validity_age_slots: int = 0
    channel_quality_confidence: float | None = None
    link_adaptation_reason: str = ""
    spatial_group_id: int | None = None
    spatial_group_size: int = 1
    spatial_compatibility_score: float = 0.0
    logical_channel_allocations: list[LogicalChannelGrant] = field(default_factory=list)
    selection_reason: str = ""
    resource_pressure_source: str = ""
    harq_preemption_cause: str | None = None
    constraint_hits: list[str] = field(default_factory=list)
    lc_multiplexing_summary: dict[str, object] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return int(self.allocated_prbs) > 0 and int(self.allocated_symbols) > 0


@dataclass
class SchedulingDecision:
    """One slot-level scheduling decision with explicit grants."""

    slot_id: int
    scheduler_mode: str
    total_prbs: int
    total_symbols: int
    grants: list[UeGrant] = field(default_factory=list)
    demands: list[UeSchedulingDemand] = field(default_factory=list)
    stage_trace: SchedulingStageTrace | None = None
    unallocated_prbs: int = 0
    unallocated_symbols: int = 0

    @property
    def scheduled_ue_ids(self) -> list[str]:
        return [grant.ue_id for grant in self.grants if grant.is_active]

    @property
    def grant_count(self) -> int:
        return len(self.scheduled_ue_ids)

    def get_grant(self, ue_id: str) -> UeGrant | None:
        for grant in self.grants:
            if str(grant.ue_id) == str(ue_id):
                return grant
        return None

