"""
Author: Yisong Fu (P22000410@njupt.edu.ucn)
File: observation.py
Description: This file defines the observation structures exposed to xApps,
rApps, and other controllers. The observation contains aggregated
system-level metrics, scheduler information, current cell configuration,
and per-UE runtime summaries including service-level attributes.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CandidateCellObservation:
    cell_id: str
    cell_type: str = "macro"
    distance_2d_m: float = 0.0
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    subcarrier_spacing_hz: float = 30e3
    rf_band: str = "sub6"
    adjacent_channel_leakage_ratio_db: float = 45.0
    adjacent_channel_selectivity_db: float = 33.0
    out_of_band_isolation_db: float = 42.0
    tx_power_dbm: float = 30.0
    connected_ue_count: int = 0
    scheduled_ue_count: int = 0
    queued_bytes: int = 0
    estimated_load: float = 0.0
    load_penalty_db: float = 0.0
    estimated_interference_db: float = -120.0
    interference_penalty_db: float = 0.0
    signal_margin_db: float = -120.0
    handover_cost_db: float = 0.0
    steering_hysteresis_db: float = 0.0
    hysteresis_adjusted_score_db: float = -100.0
    measurement_event_trigger: str = "a3_offset"
    measurement_event_offset_db: float = 0.0
    measurement_event_met: bool = False
    trigger_margin_db: float = -100.0
    steering_score_db: float = -100.0
    pathloss_db: float = 130.0
    rsrp_dbm: float = -100.0
    quality_score_db: float = -100.0
    candidate_beam_id: int = 0
    is_serving: bool = False
    is_target: bool = False


@dataclass
class CellObservation:
    cell_id: str
    cell_type: str = "macro"
    is_primary: bool = False
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    subcarrier_spacing_hz: float = 30e3
    rf_band: str = "sub6"
    adjacent_channel_leakage_ratio_db: float = 45.0
    adjacent_channel_selectivity_db: float = 33.0
    out_of_band_isolation_db: float = 42.0
    tx_power_dbm: float = 30.0
    current_codebook_id: int | None = None
    current_beam_profile_id: int | None = None
    scheduler_mode: str = "round_robin"
    scheduler_policy_hint: str | None = None
    scheduler_packing_strategy: str | None = None
    latency_bias: float | None = None
    slice_weight_adjustment: dict[str, float] = field(default_factory=dict)
    beam_candidate_preference: str | None = None
    power_cap_dbm: float | None = None
    cell_sleep_mode: str | None = None
    traffic_steering_preference: str | None = None
    mobility_bias_db: float | None = None
    handover_margin_db: float | None = None
    handover_cost_db: float | None = None
    handover_interruption_slots: int | None = None
    steering_hysteresis_db: float | None = None
    measurement_event_trigger: str | None = None
    measurement_event_offset_db: float | None = None
    connected_ue_count: int = 0
    scheduled_ue_count: int = 0
    total_queue_bytes: int = 0
    resource_grid_total_re: int = 0
    resource_grid_allocated_re: int = 0
    resource_grid_utilization: float = 0.0
    interference_penalty_db_mean: float | None = None
    inter_cell_interference_penalty_db_mean: float | None = None
    inter_cell_frequency_coupling_fraction_mean: float | None = None
    inter_cell_interference_source_count_mean: float | None = None
    active_ue_ids: List[str] = field(default_factory=list)
    scheduled_ue_ids: List[str] = field(default_factory=list)


@dataclass
class LogicalChannelObservation:
    logical_channel_id: str
    drb_id: int | None = None
    lcg_id: int | None = None
    queue_bytes: int = 0
    priority: int = 0
    packet_delay_budget_ms: float = 0.0
    slice_id: str = ""
    traffic_type: str = ""
    prioritized_bit_rate_bps: float | None = None
    pbr_credit_bytes: float | None = None
    is_scheduled: bool = False
    scheduled_bytes: int = 0
    prioritized_scheduled_bytes: int = 0
    delivered_bytes: int = 0


@dataclass
class LogicalChannelGroupObservation:
    lcg_id: int
    logical_channel_ids: List[str] = field(default_factory=list)
    drb_ids: List[int] = field(default_factory=list)
    queue_bytes: int = 0
    prioritized_queue_bytes: int = 0
    pbr_credit_bytes: float = 0.0
    is_scheduled: bool = False
    scheduled_bytes: int = 0
    prioritized_scheduled_bytes: int = 0
    delivered_bytes: int = 0
    bsr_reported_bytes: int = 0
    bsr_pending_bytes: int = 0
    bsr_age_slots: int | None = None
    last_bsr_report_slot: int | None = None
    last_bsr_trigger_slot: int | None = None
    bsr_triggered: bool = False
    bsr_trigger_reason: str | None = None


@dataclass
class UeObservation:
    """
    Stores the observation summary of one UE.
    INPUT
    ue_id: The unique identifier of the UE.
    queue_bytes: The current queued traffic in bytes.
    throughput_mbps: The current UE throughput in Mbps.
    sinr_db: The current UE SINR in dB.
    bler: The current UE BLER value.
    serving_beam_id: The current serving beam identifier.
    is_scheduled: Whether the UE is scheduled in the current slot.
    slice_id: The service slice identifier of the UE.
    traffic_type: The traffic type of the UE.
    latency_budget_ms: The latency budget of the UE in milliseconds.
    priority: The service priority of the UE.
    OUTPUT
    A UE observation object used by controllers.
    EXAMPLE OUTPUT
    UeObservation(
        ue_id="ue_0",
        queue_bytes=900,
        throughput_mbps=5.0,
        sinr_db=11.0,
        bler=0.08,
        serving_beam_id=0,
        is_scheduled=True,
        slice_id="urllc",
        traffic_type="control",
        latency_budget_ms=10.0,
        priority=3
    )
    """
    ue_id: str
    queue_bytes: int
    throughput_mbps: float
    sinr_db: float
    bler: float
    serving_beam_id: int
    is_scheduled: bool
    slice_id: str
    traffic_type: str
    latency_budget_ms: float
    priority: int
    demand_true_backlog_bytes: int | None = None
    demand_observed_backlog_bytes: int | None = None
    demand_bytes: int | None = None
    demand_hol_delay_ms: float | None = None
    demand_qos_class: str | None = None
    demand_five_qi: int | None = None
    demand_bsr_like_bytes: int | None = None
    demand_bsr_reported_bytes: int | None = None
    demand_bsr_pending_bytes: int | None = None
    demand_harq_urgency_score: float | None = None
    demand_channel_quality_confidence: float | None = None
    demand_reason: str | None = None
    logical_channel_count: int | None = None
    logical_channel_group_count: int | None = None
    logical_channel_demand_count: int | None = None
    logical_channel_group_demand_count: int | None = None
    dominant_logical_channel_id: str | None = None
    dominant_lcg_id: int | None = None
    scheduled_logical_channel_ids: List[str] = field(default_factory=list)
    scheduled_drb_ids: List[int] = field(default_factory=list)
    scheduled_lcg_ids: List[int] = field(default_factory=list)
    logical_channel_observations: List[LogicalChannelObservation] = field(default_factory=list)
    logical_channel_group_observations: List[LogicalChannelGroupObservation] = field(
        default_factory=list
    )
    serving_cell_id: str | None = None
    candidate_cell_observations: List[CandidateCellObservation] = field(default_factory=list)
    connection_state: str = "connected"
    c_rnti: int | None = None
    preferred_target_cell_id: str | None = None
    handover_pending_target_cell_id: str | None = None
    handover_started_slot: int | None = None
    last_handover_slot: int | None = None
    handover_count: int | None = None
    handover_cost_db: float | None = None
    last_handover_cost_db: float | None = None
    handover_interruption_slots: int | None = None
    interruption_remaining_slots: int | None = None
    steering_hysteresis_db: float | None = None
    measurement_event_trigger: str | None = None
    measurement_event_offset_db: float | None = None
    last_measurement_event_slot: int | None = None
    last_measurement_event_type: str | None = None
    last_measurement_event_margin_db: float | None = None
    last_trigger_reason: str | None = None
    traffic_steering_bias_db: float | None = None
    rank_indicator: int | None = None
    pmi: int | None = None
    cqi: int | None = None
    active_harq_process_count: int | None = None
    num_tx_ant: int | None = None
    num_rx_ant: int | None = None
    beamforming_rank: int | None = None
    beamforming_layer_count: int | None = None
    precoder_id: int | None = None
    combiner_id: int | None = None
    precoding_matrix_label: str | None = None
    post_eq_sinr_db: float | None = None
    hol_delay_ms: float | None = None
    rsrp_dbm: float | None = None
    reference_sinr_db: float | None = None
    data_sinr_db: float | None = None
    sinr_kind: str = 'legacy_empirical'
    post_eq_sinr_source: str = 'legacy_truth'
    precoder_match_score: float | None = None
    beam_direction_deg: float | None = None
    beamforming_gain_db: float | None = None
    grant_target_mcs: int | None = None
    grant_mcs_table_id: str | None = None
    grant_tbs_selection_method: str | None = None
    grant_dmrs_pattern_id: str | None = None
    grant_cqi_validity_age_slots: int | None = None
    grant_ri_validity_age_slots: int | None = None
    grant_pmi_validity_age_slots: int | None = None
    grant_channel_quality_confidence: float | None = None
    grant_link_adaptation_reason: str | None = None
    grant_start_prb: int | None = None
    grant_start_symbol: int | None = None
    grant_allocated_prbs: int | None = None
    grant_allocated_symbols: int | None = None
    grant_resource_block_indices: List[int] = field(default_factory=list)
    grant_allocated_symbol_indices: List[int] = field(default_factory=list)
    resource_grid_re_count: int | None = None
    resource_grid_overlap_fraction: float | None = None
    interference_penalty_db: float | None = None
    intra_cell_interference_penalty_db: float | None = None
    inter_cell_interference_dbm: float | None = None
    inter_cell_interference_penalty_db: float | None = None
    inter_cell_interference_overlap_fraction: float | None = None
    inter_cell_frequency_coupling_fraction: float | None = None
    inter_cell_interference_source_count: int | None = None
    interference_group_id: int | None = None
    spatial_group_id: int | None = None
    spatial_group_size: int | None = None
    spatial_compatibility_score: float | None = None
    spatial_multiplexing_gain_db: float | None = None
    spatial_mode: str | None = None

    # Measurement quality hints (optional; used for realism-aware controllers).
    queue_age_slots: int | None = None
    sinr_window_len: int | None = None
    sinr_effective_sample_count: int | None = None
    bler_window_sample_count: int | None = None
    throughput_window_slots: int | None = None
    last_measurement_update_slot: int | None = None
    measurement_produced_at_slot: int | None = None
    measurement_delivered_at_slot: int | None = None
    measurement_age_slots: int | None = None
    queue_delay_slots_applied: int | None = None
    queue_produced_at_slot: int | None = None
    sinr_window_slots: int | None = None
    sinr_noise_std_db_applied: float | None = None
    bler_window_tbs: int | None = None
    bler_sample_count: int | None = None
    throughput_granularity_slots: int | None = None


@dataclass
class Observation:
    """
    Stores the aggregated observation exposed to controllers.
    INPUT
    slot_id: The current slot index.
    avg_sinr_db: The average SINR in dB across all UEs (measurement view), or None when undefined.
    avg_bler: The average BLER across scheduled UEs (TX-only metric), or None when no UE is scheduled.
    total_queue_bytes: The total queued traffic in bytes.
    total_throughput_mbps: The total throughput in Mbps.
    cell_id: The serving cell identifier.
    current_codebook_id: The current active codebook identifier.
    current_beam_profile_id: The current active beam profile identifier.
    tx_power_dbm: The current transmit power in dBm.
    scheduler_mode: The current scheduler mode.
    scheduled_ue_ids: The list of scheduled UE identifiers in the current slot.
    num_scheduled_ues: The number of scheduled UEs in the current slot.
    ue_observations: The list of per-UE observation objects.
    OUTPUT
    One observation object for controller use.
    EXAMPLE OUTPUT
    Observation(
        slot_id=1,
        avg_sinr_db=10.5,
        avg_bler=0.08,
        total_queue_bytes=2700,
        total_throughput_mbps=18.0,
        cell_id="cell_0",
        current_codebook_id=0,
        current_beam_profile_id=0,
        tx_power_dbm=30.0,
        scheduler_mode="round_robin",
        scheduled_ue_ids=["ue_0"],
        num_scheduled_ues=1,
        ue_observations=[...]
    )
    """
    slot_id: int
    avg_sinr_db: float | None
    avg_bler: float | None
    total_queue_bytes: int
    total_throughput_mbps: float
    cell_id: str
    current_codebook_id: int
    current_beam_profile_id: int
    tx_power_dbm: float
    scheduler_mode: str
    runtime_mode: str = "single_active_cell"
    scheduler_packing_strategy: str | None = None
    scheduler_demand_source: str | None = None
    scheduler_demand_count: int = 0
    scheduler_true_backlog_bytes_total: int = 0
    scheduler_demand_bytes_total: int = 0
    scheduler_observed_backlog_bytes_total: int = 0
    scheduler_bsr_bytes_total: int = 0
    scheduler_low_confidence_channel_count: int = 0
    scheduler_stale_csi_count: int = 0
    scheduled_ue_ids: List[str] = field(default_factory=list)
    num_scheduled_ues: int = 0
    ue_observations: List[UeObservation] = field(default_factory=list)
    cell_observations: List[CellObservation] = field(default_factory=list)
    connected_ue_count: int = 0
    out_of_sync_ue_count: int = 0
    handover_pending_ue_count: int = 0
    handover_completed_ue_count: int = 0
    handover_interruption_ue_count: int = 0
    gnb_num_tx_ant: int | None = None
    carrier_frequency_hz: float | None = None
    bandwidth_hz: float | None = None
    subcarrier_spacing_hz: float | None = None
    cyclic_prefix_type: str | None = None
    num_ofdm_symbols_per_slot: int | None = None
    resource_grid_total_re: int | None = None
    resource_grid_allocated_re: int | None = None
    resource_grid_utilization: float | None = None
    resource_grid_n_prbs_total: int | None = None
    resource_grid_n_ofdm_symbols_per_slot: int | None = None
    resource_grid_fft_size: int | None = None
    resource_grid_overlap_fraction_mean: float | None = None
    resource_grid_mu_mimo_overlap_pair_count: int | None = None
    resource_grid_mu_mimo_group_count: int | None = None
    resource_grid_mu_mimo_ue_count: int | None = None
    resource_grid_adjacent_interference_pair_count: int | None = None
    resource_grid_interference_group_count: int | None = None
    interference_penalty_db_mean: float | None = None
    inter_cell_interference_penalty_db_mean: float | None = None
    inter_cell_frequency_coupling_fraction_mean: float | None = None
    inter_cell_interference_source_count_mean: float | None = None
    spatial_multiplexing_gain_db_mean: float | None = None

    # Scoped metrics to avoid ambiguity (scheduled-only vs all-UE).
    avg_sinr_db_scheduled: float | None = None
    avg_bler_scheduled: float | None = None
    avg_sinr_db_all: float | None = None
    avg_radio_quality_db_all: float | None = None

    # Optional measurement timing metadata (when this observation was produced / delivered).
    produced_at_slot: int | None = None
    delivered_at_slot: int | None = None

    # Measurement configuration metadata.
    measurement_produced_at_slot: int | None = None
    measurement_delivered_at_slot: int | None = None
    measurement_age_slots: int | None = None
    measurement_queue_delay_slots: int | None = None
    measurement_sinr_window_slots: int | None = None
    measurement_sinr_noise_std_db_applied: float | None = None
    measurement_bler_window_tbs: int | None = None
    measurement_throughput_granularity_slots: int | None = None
    measurement_queue_age_slots_mean: float | None = None
    measurement_sinr_effective_sample_count_mean: float | None = None
    measurement_bler_sample_count_mean: float | None = None
    measurement_throughput_sample_count_mean: float | None = None
