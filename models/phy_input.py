"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: phy_input.py
Description: This file defines the PHY input structures used by the
simulator. The simulator converts the current global state into one PHY
input object before calling the PHY adapter.
"""

from dataclasses import dataclass, field
from typing import List

from models.phy_runtime_context import PhyRuntimeContext
from models.scheduling import SchedulingDecision


@dataclass
class PhyResourceGridSlice:
    """
    Stores one explicit OFDM resource-grid slice derived from a grant.
    """

    start_prb: int = 0
    allocated_prbs: int = 0
    start_symbol: int = 0
    allocated_symbols: int = 0
    resource_block_indices: List[int] = field(default_factory=list)
    allocated_subcarrier_indices: List[int] = field(default_factory=list)
    allocated_symbol_indices: List[int] = field(default_factory=list)
    dmrs_pattern_id: str | None = None
    resource_elements: int = 0


@dataclass
class PhyUeInput:
    """
    Stores the PHY input information of one UE.
    INPUT
    ue_id: The unique identifier of the UE.
    queue_bytes: The current queued traffic in bytes.
    sinr_db: The current UE measured SINR in dB.
    radio_quality_db: The stable UE radio-quality baseline in dB.
    bler: The current UE BLER value before PHY execution.
    serving_beam_id: The current serving beam identifier.
    is_scheduled: Whether the UE is scheduled in the current slot.
    slice_id: The service slice identifier of the UE.
    traffic_type: The traffic type of the UE.
    latency_budget_ms: The latency budget of the UE in milliseconds.
    priority: The service priority.
    allocated_prbs: The granted PRB amount for the UE in the current slot.
    allocated_symbols: The granted OFDM symbol amount for the UE in the current slot.
    start_prb: The starting PRB index of the UE allocation in the slot resource grid.
    start_symbol: The starting OFDM symbol index of the UE allocation in the slot resource grid.
    resource_block_indices: The explicit PRB indices occupied by the grant.
    allocated_subcarrier_indices: The explicit subcarrier indices occupied by the grant.
    allocated_symbol_indices: The explicit OFDM symbol indices occupied by the grant.
    resource_grid_slice: The fully-expanded resource-grid slice derived from the grant.
    is_retx: Whether the grant serves a pending retransmission.
    priority_score: The scheduler-side score used to rank the UE.
    target_mcs: An optional grant-level target MCS index.
    mcs_cap: An optional grant-level upper bound on the MCS index.
    harq_process_id: The HARQ process identifier for this grant.
    link_pathloss_db: The serving-link large-scale pathloss for this UE.
    link_shadowing_db: The serving-link shadowing value for this UE.
    link_doppler_hz: The serving-link Doppler spread for this UE.
    link_delay_spread_ns: The serving-link delay spread for this UE.
    link_channel_model_type: The serving-link channel model identifier.
    link_fast_fading_seed: A repeatable seed for UE-specific small-scale fading.
    num_tx_ant: The serving gNB transmit-antenna count visible to this UE.
    num_rx_ant: The UE receive-antenna count.
    rank_indicator: The current beamforming rank indication for this UE.
    layer_count: The current transmission layer count.
    precoder_id: The current precoder identifier chosen for this UE.
    combiner_id: The current combiner identifier chosen for this UE.
    beam_direction_deg: The serving beam direction in degrees.
    beamforming_gain_db: The latest beamforming gain estimate for this UE.
    OUTPUT
    One PHY UE input object.
    EXAMPLE OUTPUT
    PhyUeInput(...)
    """
    ue_id: str
    queue_bytes: int
    sinr_db: float
    radio_quality_db: float
    bler: float
    serving_beam_id: int
    is_scheduled: bool
    slice_id: str
    traffic_type: str
    latency_budget_ms: float
    priority: int
    allocated_prbs: int = 0
    allocated_symbols: int = 0
    start_prb: int = 0
    start_symbol: int = 0
    resource_block_indices: List[int] = field(default_factory=list)
    allocated_subcarrier_indices: List[int] = field(default_factory=list)
    allocated_symbol_indices: List[int] = field(default_factory=list)
    resource_grid_slice: PhyResourceGridSlice | None = None
    tx_contract: object | None = None
    is_retx: bool = False
    requested_bytes: int = 0
    grant_bytes_budget: int = 0
    priority_score: float = 0.0
    target_mcs: int | None = None
    mcs_cap: int | None = None
    mcs_table_id: str = "qam64"
    tbs_selection_method: str = "resource_elements"
    dmrs_pattern_id: str | None = None
    harq_process_id: int | None = None
    harq_redundancy_version: int | None = None
    harq_ndi: int | None = None
    harq_retransmission_round: int = 0
    cqi_validity_age_slots: int = 0
    ri_validity_age_slots: int = 0
    pmi_validity_age_slots: int = 0
    channel_quality_confidence: float | None = None
    link_adaptation_reason: str | None = None
    spatial_group_id: int | None = None
    spatial_group_size: int = 1
    spatial_compatibility_score: float = 0.0
    link_pathloss_db: float | None = None
    link_shadowing_db: float | None = None
    link_los_state: str | None = None
    link_o2i_penetration_loss_db: float | None = None
    link_doppler_hz: float | None = None
    link_delay_spread_ns: float | None = None
    link_k_factor_db: float | None = None
    link_angle_spread_deg: float | None = None
    link_spatial_correlation_level: float | None = None
    link_channel_model_type: str | None = None
    link_fast_fading_seed: int | None = None
    link_spatial_corr_seed: int | None = None
    reported_cqi: int | None = None
    reported_ri: int | None = None
    reported_pmi: int | None = None
    num_tx_ant: int | None = None
    num_rx_ant: int | None = None
    rank_indicator: int | None = None
    layer_count: int | None = None
    precoder_id: int | None = None
    combiner_id: int | None = None
    precoder_matrix: List[List[float]] = field(default_factory=list)
    combiner_vector: List[float] = field(default_factory=list)
    precoding_matrix_label: str | None = None
    beam_direction_deg: float | None = None
    beamforming_gain_db: float | None = None
    inter_cell_interference_dbm: float = -120.0
    inter_cell_interference_penalty_db: float = 0.0
    inter_cell_interference_overlap_fraction: float = 0.0
    inter_cell_frequency_coupling_fraction: float = 0.0
    inter_cell_interference_source_count: int = 0


@dataclass
class PhyInput:
    """
    Stores the full PHY input for one slot.
    INPUT
    slot_id: The current slot index.
    sim_time_ms: The current simulation time in milliseconds.
    cell_id: The serving cell identifier.
    tx_power_dbm: The current transmit power in dBm.
    current_codebook_id: The current active codebook identifier.
    current_beam_profile_id: The current active beam profile identifier.
    scheduled_ue_ids: The scheduled UE identifiers in the current slot.
    ue_inputs: The list of per-UE PHY input objects.
    runtime_context: The PHY runtime context object.
    scheduling_decision: The grant-level scheduling decision for the slot.
    OUTPUT
    One PHY input object for the current slot.
    EXAMPLE OUTPUT
    PhyInput(...)
    """
    slot_id: int
    sim_time_ms: float
    cell_id: str
    tx_power_dbm: float
    current_codebook_id: int
    current_beam_profile_id: int
    scheduled_ue_ids: List[str] = field(default_factory=list)
    ue_inputs: List[PhyUeInput] = field(default_factory=list)
    runtime_context: PhyRuntimeContext | None = None
    scheduling_decision: SchedulingDecision | None = None


