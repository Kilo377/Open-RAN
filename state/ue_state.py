"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: ue_state.py
Description: This file defines the UE runtime state used by the simulator.
The UE state stores traffic, radio, service, and HARQ-related information
for one UE. It distinguishes stable radio-quality baseline from current
slot-level measured SINR.
"""

from dataclasses import dataclass, field

from state.harq_state import HarqState
from state.logical_channel_group_state import LogicalChannelGroupState
from state.logical_channel_state import LogicalChannelState


@dataclass
class UeState:
    """
    Stores the runtime state of one UE.
    INPUT
    ue_id: The unique identifier of the UE.
    queue_bytes: The queued traffic volume in bytes.
    throughput_mbps: The current throughput in Mbps.
    sinr_db: The current slot-level measured SINR in dB.
    radio_quality_db: The stable radio-quality baseline in dB.
    bler: The current BLER value.
    serving_beam_id: The current serving beam identifier.
    slice_id: The slice identifier of the UE.
    traffic_type: The traffic type of the UE.
    latency_budget_ms: The latency budget in milliseconds.
    priority: The service priority.
    harq: The HARQ runtime state of the UE.
    OUTPUT
    One UE state object.
    EXAMPLE OUTPUT
    UeState(
        ue_id="ue_0",
        queue_bytes=1000,
        throughput_mbps=0.0,
        sinr_db=10.0,
        radio_quality_db=9.0,
        bler=0.1,
        serving_beam_id=0,
        slice_id="embb",
        traffic_type="video",
        latency_budget_ms=100.0,
        priority=1,
        harq=HarqState()
    )
    """
    ue_id: str
    queue_bytes: int
    throughput_mbps: float
    sinr_db: float
    radio_quality_db: float
    bler: float
    serving_beam_id: int
    slice_id: str
    traffic_type: str
    latency_budget_ms: float
    priority: int
    harq: HarqState = field(default_factory=HarqState)
    logical_channels: list[LogicalChannelState] = field(default_factory=list)
    logical_channel_groups: list[LogicalChannelGroupState] = field(default_factory=list)

    # Slot-level fields updated by StateUpdater / PHY.
    is_scheduled: bool = False
    allocated_prbs: int = 0
    allocated_symbols: int = 0
    start_prb: int = 0
    start_symbol: int = 0
    resource_block_indices: list[int] = field(default_factory=list)
    allocated_subcarrier_indices: list[int] = field(default_factory=list)
    allocated_symbol_indices: list[int] = field(default_factory=list)
    is_retx: bool = False
    priority_score: float = 0.0
    pre_eq_sinr_db: float = 0.0
    post_eq_sinr_db: float = 0.0
    channel_rank_limit: int = 1
    precoder_match_score: float = 1.0
    mcs_index: int = 0
    transport_block_size_bytes: int = 0
    grant_bytes_budget: int = 0
    grant_target_mcs: int | None = None
    grant_mcs_table_id: str = "qam64"
    grant_tbs_selection_method: str = "resource_elements"
    grant_dmrs_pattern_id: str | None = None
    grant_cqi_validity_age_slots: int = 0
    grant_ri_validity_age_slots: int = 0
    grant_pmi_validity_age_slots: int = 0
    grant_channel_quality_confidence: float = 1.0
    grant_link_adaptation_reason: str = ""
    scheduled_bytes: int = 0
    delivered_bytes: int = 0
    new_tx_bytes: int = 0
    retx_tx_bytes: int = 0
    delivered_new_bytes: int = 0
    delivered_retx_bytes: int = 0
    ack: bool = False
    cqi: int = 0
    rank: int = 1
    pmi: int = 0
    ri: int = 1
    resource_grid_re_count: int = 0
    resource_grid_overlap_fraction: float = 0.0
    interference_penalty_db: float = 0.0
    intra_cell_interference_penalty_db: float = 0.0
    inter_cell_interference_dbm: float = -120.0
    inter_cell_interference_penalty_db: float = 0.0
    inter_cell_interference_overlap_fraction: float = 0.0
    inter_cell_frequency_coupling_fraction: float = 0.0
    inter_cell_interference_source_count: int = 0
    spatial_group_id: int | None = None
    spatial_group_size: int = 1
    spatial_compatibility_score: float = 0.0
    spatial_multiplexing_gain_db: float = 0.0
    spatial_mode: str = "su_siso"
    scheduled_logical_channel_ids: list[str] = field(default_factory=list)
    scheduled_drb_ids: list[int] = field(default_factory=list)
    scheduled_lcg_ids: list[int] = field(default_factory=list)


